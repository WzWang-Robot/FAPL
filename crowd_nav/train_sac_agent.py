#!/usr/bin/env python3
from collections import deque
import csv
import numpy as np
from reward_model import RewardModel
from replay_buffer import ReplayBuffer
import os
import gym
import configparser
import torch
from logger import Logger
import utils_a as utils
import hydra
import time
from crowd_sim.envs.utils.robot import Robot


class Active_Learning:
    def __init__(self, cfg):
        self.name_step = '4000'
        self.buffer_dir = "/home/dinosaur/CrowdNav/crowd_nav/buffer_data/" + self.name_step
        self.buffer_name = 'actions.csv'
        self.buffer_path = self.buffer_dir + self.buffer_name
        self.device = torch.device(cfg.device)
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')
        self.cfg = cfg
        self.agent_state_dim = 9
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent=cfg.agent.name)
        self.seed_sample_flag = True
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        env_config = configparser.RawConfigParser()
        env_config.read('/home/dinosaur/CrowdNav/crowd_nav/configs/env.config')
        self.human_num = env_config.getint('sim', 'human_num')
        self.action_dim = 2
        cfg.agent.params.obs_dim = 5 * self.human_num + self.agent_state_dim
        cfg.agent.params.action_dim = self.action_dim
        cfg.agent.params.action_range = [
            float(-5), float(5)
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)

        self.replay_buffer = ReplayBuffer(
            (5 * self.human_num + self.agent_state_dim,), (self.action_dim,),
            int(cfg.replay_buffer_capacity),
            self.device)
        if os.path.isfile(self.buffer_path):
            print("buffer ok")
            self.replay_buffer.load(self.buffer_dir)
        else:
            print("no buffer exist")
        # if os.path.isfile('%s/actor_%s.pt' % (self.work_dir, self.name_step)):
        #     print("agent ok")
        #     self.seed_sample_flag = False
        #     self.agent.load(self.work_dir, self.name_step)

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

    def run(self):
        # self.agent.reset_critic()
        # self.agent.update_after_reset(
        #     self.replay_buffer, self.logger, self.step,
        #     gradient_update=self.cfg.reset_update,
        #     policy_update=True)

        if os.path.isfile('%s/actor_%s.pt' % (self.work_dir, self.name_step)):
            print("agent ok")
            self.seed_sample_flag = False
            self.agent.load(self.work_dir, self.name_step)
        while self.step < self.cfg.num_reward_train_steps:
            if self.total_feedback < self.cfg.max_feedback:
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)
            print(self.step)
            self.step += 1
        self.agent.save(self.work_dir, self.cfg.num_train_steps)


@hydra.main(config_path='configs/train_FAPL.yaml', strict=True)
def main(cfg):
    workspace = Active_Learning(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
