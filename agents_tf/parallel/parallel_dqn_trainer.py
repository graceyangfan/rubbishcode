import numpy as np

from typing import Callable
from multiprocessing import Process, Queue
from zenquant.env.backtest_env import BacktestEnv 

class ParallelDQNTrainer(Process):

    def __init__(self,
                 agent: 'ParallelDQNAgent',
                 create_env: Callable[[None], 'BacktestEnv'],
                 memory_queue: Queue,
                 model_update_queue: Queue,
                 done_queue: Queue,
                 n_steps: int,
                 n_episodes: int,
                 eps_end: int = 0.05,
                 eps_start: int = 0.99,
                 eps_decay_steps: int = 2000,
                 update_target_every: int = 2):
        super().__init__()

        self.agent = agent
        self.env = create_env()
        self.memory_queue = memory_queue
        self.model_update_queue = model_update_queue
        self.done_queue = done_queue
        self.n_steps = n_steps or np.iinfo(np.int32).max
        self.n_episodes = n_episodes or np.iinfo(np.int32).max
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.update_target_every = update_target_every

        self.env.agent_id = self.agent.id

    def run(self):
        episode = 0
        steps_done = 0
        total_reward = 0
        stop_training = False

        while episode < self.n_episodes and not stop_training:
            if self.model_update_queue.qsize() > 0:
                while self.model_update_queue.qsize() > 0:
                    model = self.model_update_queue.get()

                self.agent.model.update_networks(model)

            state = self.env.reset()
            done = False

            print('====      EPISODE ID ({}/{}): {}      ===='.format(episode + 1,
                                                                      self.n_episodes,
                                                                      self.env.episode_id))

            while not done:
                threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                    np.exp(-steps_done / self.eps_decay_steps)
                action = self.agent.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                self.memory_queue.put((state, action, reward, next_state, done))

                state = next_state
                total_reward += reward
                steps_done += 1

                if self.n_steps and steps_done >= self.n_steps:
                    stop_training = True
                    done = True

                if steps_done % self.update_target_every == 0:
                    self.agent.update_target_network()

            episode += 1

        mean_reward = total_reward / steps_done

        self.done_queue.put(mean_reward)