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


class Active_Learning:
    def __init__(self, cfg):
        self.name_step = '4000'
        self.buffer_dir = "/home/dinosaur/CrowdNav/crowd_nav/buffer_data/" + self.name_step
        self.buffer_name = 'actions.csv'
        self.buffer_path = self.buffer_dir + self.buffer_name
        self.reward_data_path = "/home/dinosaur/CrowdNav/crowd_nav/reward_model_data/"
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
        self.env = gym.make('CrowdSim-v0')
        self.env.configure(env_config)
        self.human_num = env_config.getint('sim', 'human_num')
        self.action_dim = 2
        cfg.agent.params.obs_dim = 5 * self.human_num + self.agent_state_dim
        cfg.agent.params.action_dim = self.action_dim
        cfg.agent.params.action_range = [
            float(-5), float(5)
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.env.set_robot(self.agent)

        self.replay_buffer = ReplayBuffer(
            (5 * self.human_num + self.agent_state_dim,), (self.action_dim,),
            int(cfg.replay_buffer_capacity),
            self.device)
        if os.path.isfile(self.buffer_path):
            print("buffer ok")
            self.replay_buffer.load(self.buffer_dir)
        else:
            print("no buffer exist")
        if os.path.isfile('%s/actor_%s.pt' % (self.work_dir, self.name_step)):
            print("agent ok")
            self.seed_sample_flag = False
            self.agent.load(self.work_dir, self.name_step)
        # self.agent.load()

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        self.reward_model = RewardModel(
            5 * self.human_num + self.agent_state_dim, self.action_dim,
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal)
        if os.path.isfile('%s/reward_model_%s_0.pt' % (self.work_dir, self.name_step)):
            print("reward model parameter ok")
            self.reward_model.load(self.work_dir, self.name_step)
        else:
            print("no reward model weight exist!")
        if os.path.isfile('%sinputs.csv' % (self.reward_data_path)):
            print("reward model data ok")
            self.reward_model.load_reward_data(self.reward_data_path)
        else:
            print("no reward data exist!")

    def learn_reward(self, first_flag=0):

        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling(self.env)
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling(self.env)
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)

                if total_acc > 0.97:
                    break;

        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0

        start_time = time.time()

        interact_count = 0
        avg_train_true_return = deque([], maxlen=10)
        episode_reward_path = self.reward_data_path + "avg_train_true_return.csv"
        with open(episode_reward_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for data in row:
                    avg_train_true_return.append(float(data))

        # update margin --> not necessary / will be updated soon
        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / 200)
        self.reward_model.set_teacher_thres_skip(new_margin)
        self.reward_model.set_teacher_thres_equal(new_margin)
        self.learn_reward(first_flag=1)
        self.replay_buffer.relabel_with_predictor(self.reward_model)
        self.agent.reset_critic()
        self.agent.update_after_reset(
            self.replay_buffer, self.logger, self.step,
            gradient_update=self.cfg.reset_update,
            policy_update=True)
        while self.step < self.cfg.num_reward_train_steps:
            if self.total_feedback < self.cfg.max_feedback:
                # if interact_count == self.cfg.num_interact:
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)

                # update margin --> not necessary / will be updated soon
                # new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / 200)
                self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)

                # corner case: new total feed > max feed
                if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                    self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                self.learn_reward()
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                # interact_count = 0
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step,
                                            gradient_update=1, K=self.cfg.topK)
            self.step += 1
            # interact_count += 1
        self.agent.save(self.work_dir, self.cfg.num_train_steps)
        self.reward_model.save(self.work_dir, self.cfg.num_train_steps)
        self.replay_buffer.save_reward(self.buffer_dir)
        # self.reward_model.save_reward_data()


@hydra.main(config_path='configs/train_FAPL.yaml', strict=True)
def main(cfg):
    workspace = Active_Learning(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
