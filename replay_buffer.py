import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler

def choose(traj_list, max_traj):
    random.shuffle(traj_list)
    return (traj_list if max_traj < 0 else traj_list[:max_traj])

def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def load_episode(fn, action_dim=None, action_index=None):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        if action_dim is not None:
            action = np.zeros((episode['action'].shape[0], action_dim), dtype=episode['action'].dtype)
            action[:, action_index[0]:action_index[1]] = episode['action']
            episode['action'] = action
        return episode


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, num_workers, nstep,  
                 fetch_every, n_traj=5,
                 window_size=3, rank=None, world_size=None, 
                 action_indices=None, bc=False):
        self._replay_dir = replay_dir if type(replay_dir)== list else [replay_dir]
        self._size = 0
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self.window_size = window_size
        self.action_indices = action_indices
        self._n_traj = n_traj
        if self.action_indices is None:
            self._action_dim = None
        else:
            self._action_dim = action_indices[-1][-1]
        self.rank = rank
        self.world_size = world_size
        self.bc = bc
        print('Loading Data into CPU Memory')
        self._preload()

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]
    
    def __len__(self):
        return self._size
    
    def _store_episode(self, eps_fn, index=None):
        if index is None:
            episode = load_episode(eps_fn)
        else:
            episode = load_episode(eps_fn, self._action_dim, self.action_indices[index])
        eps_len = episode_len(episode)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        return True
    
    def _preload(self):
        index = 0
        for replay_dir in self._replay_dir:
            eps_fns = choose(sorted(replay_dir.glob('*.npz'), reverse=True), self._n_traj)
            if self.rank is not None:
                for eps_fn in eps_fns:
                    eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
                    if eps_idx % self.world_size == self.rank:
                        if self.action_indices is not None:
                            self._store_episode(eps_fn, index)
                        else:
                            self._store_episode(eps_fn)
            else:
                for eps_fn in eps_fns:
                    if self.action_indices is not None:
                            self._store_episode(eps_fn, index)
                    else:
                        self._store_episode(eps_fn)
                    
            index += 1
        
    def _sample(self):
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        n_step = self._nstep + self.window_size
        idx = np.random.randint(0, episode_len(episode) - n_step + 1) + 1
        if self.bc:
            obs = episode['observation'][idx - 1]
            action = episode['action'][idx]
            return (obs, action)
        else:
            obs = episode['observation'][idx - 1]
            next_obs = episode['observation'][idx + self._nstep - 1]
            ### sample negative pair of r_next_obs from its nearby states
            if np.random.uniform(-1,1) > 0:
                index = (idx + self._nstep - 1) + (np.random.randint(self.window_size)+1) 
            else:
                index = (idx + self._nstep - 1) - (np.random.randint(self.window_size)+1) 
            neg_next_obs = episode['observation'][index]
            
#             ### Sample negative example
#             neg_sample_window_left = max(0, idx + self._nstep - 1 - self.window_size)
#             neg_sample_window_right = min(episode_len(episode), idx + self._nstep + self.window_size)
#             neg_index = np.random.randint(neg_sample_window_left, neg_sample_window_right)
#             neg_next_obs = episode['observation'][neg_index]

            action = episode['action'][idx]
            action_seq = np.concatenate([episode['action'][idx+i] for i in range(self._nstep)])
            return (obs, action_seq, next_obs, neg_next_obs)
            
    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, batch_size, num_workers,
                       nstep, n_traj=5, window_size=3, 
                       action_indices=None, rank=None, 
                       world_size=None, bc=False):
    
    iterable = ReplayBuffer(replay_dir,
                            num_workers,
                            nstep,
                            fetch_every=1000,
                            window_size=window_size,
                            action_indices=action_indices,
                            rank=rank,
                            n_traj=n_traj, 
                            world_size=world_size,
                            bc=bc)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn)
    return loader






