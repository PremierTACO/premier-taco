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

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.eval_env.observation_spec(),
                                self.eval_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, offline=True)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        
        self.replay_loader = make_replay_loader(
            [Path(self.cfg.offline_data_dir)], self.cfg.batch_size, 
            self.cfg.replay_buffer_num_workers,
            self.cfg.nstep, n_traj=self.cfg.n_traj,
            bc=True)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


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

    def eval(self):
        step, episode, total_reward, success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_step}.mp4')
        
        with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
            log('eval_reward', total_reward / episode)
            log('step', self.global_step)

    def train(self):
        metrics = None
        while self.global_step < self.cfg.num_train_steps:
            if self.global_step%1000 == 0:
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_step,
                                                      ty='train') as log:
                        log('total_time', total_time)
                        log('step', self.global_step)
                # # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_encoder()

            # # try to evaluate
            if self.global_step%self.cfg.eval_freq==0:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_step)
                self.eval()
                            
            self._global_step += 1
            metrics = self.agent.update(self.replay_iter, self.global_step)
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            
        
            
    def load_encoder(self, encoder_dir):
        encoder_dir = Path(encoder_dir)
        with encoder_dir.open('rb') as f:
            payload = torch.load(f, map_location=torch.device('cuda'))
            self.__dict__['agent'].load_encoder = True
            self.__dict__['agent'].encoder.load_state_dict(payload)
            if self.cfg.freeze_encoder:
                self.__dict__['agent'].encoder.eval()
                utils.set_requires_grad(self.__dict__['agent'].encoder)
            
    def save_encoder(self):
        snapshot = self.work_dir / "encoder.pt"
        payload = self.__dict__['agent'].encoder.state_dict()
        with snapshot.open('wb') as f:
            torch.save(payload, f)

@hydra.main(config_path='cfgs', config_name='bc_config')
def main(cfg):
    from train_bc import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    if cfg.encoder_dir != None:
        workspace.load_encoder(cfg.encoder_dir)
    workspace.train()


if __name__ == '__main__':
    main()
