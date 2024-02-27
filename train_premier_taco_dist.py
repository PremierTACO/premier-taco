import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import hydra
import numpy as np
import torch
from dm_env import specs
import dmc
import utils
from logger_offline import Logger
from replay_buffer import make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


torch.backends.cudnn.benchmark = True


def ddp_setup(rank, world_size, port):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def make_agent(obs_spec, action_spec, rank, cfg):
    cfg.obs_shape = obs_spec
    cfg.action_shapes = action_spec
    cfg.device = 'cuda:{}'.format(rank)
    return hydra.utils.instantiate(cfg)

class Workspace:
    def __init__(self, cfg, rank, world_size):
        self.work_dir = Path.cwd()
        self.rank = rank
        self.world_size = world_size
        if rank == 0:
            print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.setup()
        
        self.agent = make_agent((3*self.cfg.frame_stack, 84,84),
                                self.a_dim,
                                rank,
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, offline=True)
        # create envs
        self.a_dim, self.replay_loaders, index = 0, [], 0
        self.action_indices, domain, start_idx = [], None, 0
        offline_data_dirs = []
        for task_name in self.cfg.task_names:
            
            env = dmc.make(task_name, self.cfg.frame_stack,
                            self.cfg.action_repeat, self.cfg.seed)
            action_dim = env.action_spec().shape[0]
            curr_domain, _ = task_name.split('_', 1)
            self.action_indices.append((start_idx, start_idx + action_dim))
            
            ### If the pretraining data comes from a new domain, increment action dimension
            if curr_domain != domain:
                start_idx += action_dim
                domain = curr_domain
                self.a_dim += env.action_spec().shape[0]
            offline_data_dir = '{}/{}_replay'.format(self.cfg.offline_data_dir, task_name)
            offline_data_dirs.append(Path(offline_data_dir))
            index += 1
            
        self.replay_loader = make_replay_loader(
            offline_data_dirs, self.cfg.batch_size//self.world_size, 
            self.cfg.replay_buffer_num_workers,
            self.cfg.nstep, n_traj=self.cfg.n_traj, 
            window_size=self.cfg.window_size, action_indices = self.action_indices,
            rank=self.rank, world_size=self.world_size
        )
        self._replay_iter = None

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def train(self):
        metrics = None
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%100 == 0:
                if metrics is not None and self.rank == 0:
                    # log stats
                    print('Premier_TACO_LOSS:{}'.format(metrics['premier_taco_loss']))
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                        log('total_time', total_time)
                        log('step', self.global_step)

                # save encoder
                if self.cfg.save_snapshot:
                    if self.global_step%10000 == 0 and self.rank==0:
                        self.save_encoder()

            self._global_step += 1
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')
        
    
    def save_encoder(self):
        counter = (self.global_step // 10000) + 1
        snapshot = self.work_dir / "encoder_{}.pt".format(counter)
        
        payload = self.__dict__['agent'].encoder.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)



RANK = None
WORLD_SIZE = None

@hydra.main(config_path='cfgs', config_name='premier_taco_config')
def main(cfg):
    global RANK, WORLD_SIZE, BARRIER
    ddp_setup(RANK, WORLD_SIZE, cfg.port)
    from train_premier_taco_dist import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg, RANK, WORLD_SIZE)
    print('Process:{} Waiting for other processes to finish setup'.format(RANK), flush=True)
    BARRIER += 1
    while BARRIER.item() < WORLD_SIZE:
        continue
    print('Process:{} Start Training'.format(RANK), flush=True)
    workspace.train()
    destroy_process_group()

def wrapper(rank, world_size, barrier):
    global RANK, WORLD_SIZE, BARRIER
    RANK = rank
    WORLD_SIZE = world_size
    BARRIER = barrier
    main()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    barrier = torch.zeros(1)
    barrier.share_memory_()
    mp.spawn(wrapper, args=(world_size,barrier), nprocs=world_size)
