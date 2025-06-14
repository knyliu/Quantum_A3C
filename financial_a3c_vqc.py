import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import os
from utils import v_wrap, set_init, push_and_pull, record
from plot_functions import full_plotting
from shared_adam import SharedAdam
from custom_torch_vqc import TorchVQC
from financial_env import FinancialTradingEnv, generate_price_series

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 1000

prices = generate_price_series(length=200)
env = FinancialTradingEnv(prices)

latent_dim = 8
n_qubits = latent_dim
q_depth = 2
n_class = latent_dim

N_S = env.observation_space.shape[0]
N_A = env.action_space.n

class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.pi1 = nn.Linear(s_dim, latent_dim)
        self.pi_vqc = TorchVQC(vqc_depth=q_depth, n_qubits=n_qubits, n_class=n_class)
        self.pi2 = nn.Linear(latent_dim, a_dim)

        self.v1 = nn.Linear(s_dim, latent_dim)
        self.v_vqc = TorchVQC(vqc_depth=q_depth, n_qubits=n_qubits, n_class=n_class)
        self.v2 = nn.Linear(latent_dim, 1)

        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        pi1 = torch.tanh(self.pi_vqc(pi1))
        logits = self.pi2(pi1)

        v1 = torch.tanh(self.v1(x))
        v1 = torch.tanh(self.v_vqc(v1))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super().__init__()
        self.name = f'w{name:02d}'
        self.g_ep = global_ep
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue
        self.gnet = gnet
        self.opt = opt
        self.lnet = Net(N_S, N_A)
        self.env = FinancialTradingEnv(prices)

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            epi_step = 1
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                epi_step += 1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done or epi_step >= len(prices):
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)

if __name__ == "__main__":
    gnet = Net(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(min(mp.cpu_count(), 8))]
    [w.start() for w in workers]

    res = []
    while True:
        r = res_queue.get()
        if len(res) < MAX_EP and (r is not None):
            res.append(r)
        elif res_queue.empty():
            break
    [w.join() for w in workers]

    full_plotting(_fileTitle="VQC_A3C_Financial", _trainingLength=len(res), _currentRewardList=res)
