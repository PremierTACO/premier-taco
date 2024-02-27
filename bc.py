import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import itertools
import torchvision.transforms as T
    
class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                 nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        self.repr_dim = feature_dim
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return self.trunk(h)

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, 
                 hidden_dim):
        super().__init__()
        
        
        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs):
        return self.policy(obs)


class BCAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, mse):
        self.device = device
        self.mse = mse
        
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        

        parameters = itertools.chain(self.encoder.parameters(),
                                     self.actor.parameters()
                                     )
        self.load_encoder = False
        self.opt         = torch.optim.Adam(parameters, lr=lr)
        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.encoder.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        action = self.actor(obs)
        action = torch.clamp(action, -1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_bc(self, obs, b_act):
        metrics = dict()
        action = self.actor(obs)
        
        ### Choose mse or l1 loss
        if self.mse:
            loss = F.mse_loss(action, b_act)
        else:
            loss = F.l1_loss(action, b_act)
        
        ### Optimize policy
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics['bc_loss'] = loss.item()

        return metrics
        
    
    def update(self, replay_iter, step):
        metrics = dict()
        
        batch = next(replay_iter)
        obs, action, = utils.to_torch(batch, self.device)
        
        metrics.update(
            self.update_bc(self.encoder(obs.float()), action))
        
        return metrics