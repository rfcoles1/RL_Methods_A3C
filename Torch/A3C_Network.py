import torch 
import torch.nn as nn

from A3C_Config import Config
from A3C_Helper import *

config = Config()

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.p1 = nn.Linear(config.s_size, 100)
        self.p2 = nn.Linear(100,config.a_size)
        self.v1 = nn.Linear(config.s_size, 100)
        self.v2 = nn.Linear(100, 1)
        set_init([self.p1, self.p2, self.v1, self.v2])
        
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        p1 = nn.functional.relu(self.p1(x))
        logits = self.p2(p1)
        v1 = nn.functional.relu(self.v1(x))
        values = self.v2(v1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        logits, _ = self.forward(s)
        prob = nn.functional.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = nn.functional.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss



