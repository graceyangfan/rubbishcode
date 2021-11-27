import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DataParallelPlugin, DDP2Plugin
from torch import Tensor, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from zenquant.agents.common.experience_source import Experience, ExperienceSourceDataset
from zenquant.agents.common.loss import dqn_loss
from zenquant.agents.common.agent import ValueAgent
from zenquant.agents.common.memory import MultiStepBuffer
from zenquant.agents.common.networks import CNN
from vnpy.trader.constant import (
    Status,
    Direction,
    Offset,
    Exchange
)


from zenquant.env.backtest_env import BacktestEnv 

class DQN(LightningModule):
    def __init__(
        self,
        env: 'BacktestEnv',
        test_env: Optional['BacktestEnv'] = None,
        **kwargs,
    ):
        self.eps_start = kwargs.get("eps_start",1.0)
        self.eps_end = kwargs.get("eps_end",0.02)
        self.eps_last_frame = kwargs.get("eps_last_frame",150000) 
        self.sync_rate = kwargs.get("sync_rate",1000)
        self.gamma = kwargs.get("gamma",0.99)
        self.learning_rate = kwargs.get("learning_rate",0.001) 
        self.batch_size = kwargs.get("batch_size",32) 
        self.replay_size = kwargs.get("replay_size",10000) 
        self.warm_start_size =  kwargs.get("warm_start_size",1000) ##large than batch size 
        self.avg_reward_len = kwargs.get("avg_reward_len",100) 
        self.min_episode_reward =  kwargs.get("min_episode_reward",-21)
        self.seed = kwargs.get("seed",42)
        self.batches_per_epoch = kwargs.get("batches_per_epoch",1000) 
        self.n_steps = kwargs.get("n_steps",1) 

        super().__init__()
        self.env = env 
        self.env.seed(self.seed)
        self.test_env = test_env 
        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n
        # Model Attributes
        self.buffer = None
        self.dataset = None

        self.net = None
        self.target_net = None

        self.build_networks()

        self.agent = ValueAgent(
            self.net,
            eps_start=self.eps_start,
            eps_end=self.eps_end,
            eps_frames=self.eps_last_frame,
        )

        self.save_hyperparameters()
        # Metrics
        self.total_episode_steps = [0]
        self.total_rewards = [0]
        self.done_episodes = 0
        self.total_steps = 0

        for _ in range(self.avg_reward_len):
            self.total_rewards.append(torch.tensor(self.min_episode_reward, device=self.device))

        self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
        ##初始状态
        self.state = self.env.reset()

        self.action_idx_list = self.get_valid_action_space()
    def get_valid_action_space(self):
        actions = self.env.action.actions
        #pos_list=[[True,True],[True,False],[False,True],[False,False],no open]
        n= self.n_actions
        action_idx_list = [list(range(n)),list(range(n)),list(range(n)),list(range(n)),list(range(n))]
        for idx,item in enumerate(actions):
            ##无空不可平空
            if item and item[0] == Direction.LONG and item[1] == Offset.CLOSE:
                action_idx_list[1].remove(idx)
                action_idx_list[3].remove(idx)
            ##无多不可平多
            if item and item[0] == Direction.SHORT and item[1] == Offset.CLOSE:
                action_idx_list[2].remove(idx)
                action_idx_list[3].remove(idx)
            if item and item[1] == Offset.OPEN:
                action_idx_list[4].remove(idx)
        return action_idx_list
    def get_current_action_space(self):
        if self.env.portfolio.occupy_rate > self.env.action.limit_order_margin_rate:
            return self.action_idx_list[4]
        elif self.env.portfolio.long_pos > self.env.min_volume:
            if self.env.portfolio.short_pos > self.env.min_volume:
                return self.action_idx_list[0]
            else:
                return self.action_idx_list[1]
        else:
            if self.env.portfolio.short_pos > self.env.min_volume:
                return self.action_idx_list[2]
            else:
                return self.action_idx_list[3]

    def run_n_episodes(self, env, n_epsiodes: int = 1, epsilon: float = 1.0) -> List[int]:
        """Carries out N episodes of the environment with the current agent.
        Args:
            env: environment to use, either train environment or test environment
            n_epsiodes: number of episodes to run
            epsilon: epsilon value for DQN agent
        """
        total_rewards = []

        for _ in range(n_epsiodes):
            episode_state = env.reset()
            done = False
            episode_reward = 0

            while not done:
                self.agent.epsilon = epsilon
                action_space = self.get_current_action_space()
                action = self.agent(episode_state,action_space, self.device)
                next_state, reward, done, _ = env.step(action[0])
                episode_state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        return total_rewards
    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience."""
        if warm_start > 0:
            self.state = self.env.reset()

            for _ in range(warm_start):
                self.agent.epsilon = 1.0
                action_space = self.get_current_action_space()
                action = self.agent(self.state,action_space, self.device)
                next_state, reward, done, _ = self.env.step(action[0])
                exp = Experience(state=self.state, action=action[0], reward=reward, done=done, new_state=next_state)
                self.buffer.append(exp)
                self.state = next_state

                if done:
                    self.state = self.env.reset()
    def build_networks(self) -> None:
        """Initializes the DQN train and target networks."""
        ##input feature_dim = input_channel,size
        self.net = CNN(self.obs_shape[1],self.obs_shape[0],(self.obs_shape[1]+self.n_actions)//2, self.n_actions)
        self.target_net = CNN(self.obs_shape[1],self.obs_shape[0],(self.obs_shape[1]+self.n_actions)//2, self.n_actions)
    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.
        Args:
            x: environment state
        Returns:
            q values
        """
        output = self.net(x)
        return output
    def train_batch(
        self,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Contains the logic for generating a new batch of data to be passed to the DataLoader.
        Returns:
            yields a Experience tuple containing the state, action, reward, done and next_state.
        """
        episode_reward = 0
        episode_steps = 0

        while True:
            self.total_steps += 1
            action_space = self.get_current_action_space()
            action = self.agent(self.state, action_space, self.device)

            next_state, r, is_done, _ = self.env.step(action[0])

            episode_reward += r
            episode_steps += 1

            exp = Experience(state=self.state, action=action[0], reward=r, done=is_done, new_state=next_state)

            self.agent.update_epsilon(self.global_step)
            self.buffer.append(exp)
            self.state = next_state

            if is_done:
                print("epsiode end")
                print(self.env.portfolio.net_capital)
                self.done_episodes += 1
                self.total_rewards.append(episode_reward)
                self.total_episode_steps.append(episode_steps)
                self.avg_rewards = float(np.mean(self.total_rewards[-self.avg_reward_len :]))
                self.state = self.env.reset()
                episode_steps = 0
                episode_reward = 0

            states, actions, rewards, dones, new_states = self.buffer.sample(self.batch_size)

            for idx, _ in enumerate(dones):
                yield states[idx], actions[idx], rewards[idx], dones[idx], new_states[idx]

            # Simulates epochs
            if self.total_steps % self.batches_per_epoch == 0:
                break

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.
        Args:
            batch: current mini batch of replay data
            _: batch number, not used
        Returns:
            Training loss and log metrics
        """

        # calculates training loss
        loss = dqn_loss(batch, self.net, self.target_net, self.gamma,torch.nn.SmoothL1Loss())

        if self._use_dp_or_ddp2(self.trainer):
            loss = loss.unsqueeze(0)

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "total_reward": self.total_rewards[-1],
                "avg_reward": self.avg_rewards,
                "train_loss": loss,
                "episodes": self.done_episodes,
                "episode_steps": self.total_episode_steps[-1],
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_rewards,
            }
        )

    def test_step(self, *args, **kwargs) -> Dict[str, Tensor]:
        """Evaluate the agent for 10 episodes."""
        test_reward = self.run_n_episodes(self.test_env, 1, 0)
        avg_reward = sum(test_reward) / len(test_reward)
        return {"test_reward": avg_reward}

    def test_epoch_end(self, outputs) -> Dict[str, Tensor]:
        """Log the avg of the test results."""
        rewards = [x["test_reward"] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        self.log("avg_test_reward", avg_reward)
        return {"avg_test_reward": avg_reward}

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        self.buffer = MultiStepBuffer(self.replay_size, self.n_steps)
        self.populate(self.warm_start_size)

        self.dataset = ExperienceSourceDataset(self.train_batch)
        return DataLoader(dataset=self.dataset, batch_size=self.batch_size)

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader."""
        return self._dataloader()
    @staticmethod
    def _use_dp_or_ddp2(trainer: Trainer) -> bool:
        return isinstance(trainer.training_type_plugin, (DataParallelPlugin, DDP2Plugin))
