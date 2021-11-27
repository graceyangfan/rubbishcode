import random
import numpy as np
import tensorflow as tf
from collections import namedtuple
from vnpy.trader.constant import (
    Status,
    Direction,
    Offset,
    Exchange
)

from zenquant.agents_tf.agent import Agent
from zenquant.agents_tf.replay_memory import  ReplayMemory
from datetime import datetime
from zenquant.env.backtest_env import BacktestEnv 


DQNTransition = namedtuple('DQNTransition', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNAgent(Agent):
    """
    """
    def __init__(self,
                 env: 'BacktestEnv',
                 policy_network: tf.keras.Model = None):
        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.init_observation.shape

        self.policy_network = policy_network or self._build_policy_network()

        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False

        self.env.agent_id = self.id
        self.action_idx_list = self.get_valid_action_space()
    def _build_policy_network(self):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.observation_shape),
            tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_actions, activation="sigmoid"),
            tf.keras.layers.Dense(self.n_actions, activation="softmax")
        ])

        return network
    def restore(self, path: str, **kwargs):
        self.policy_network = tf.keras.models.load_model(path)
        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False
    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
        else:
            filename = "policy_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"

        self.policy_network.save(path + filename)
    def get_valid_action_space(self):
        actions = self.env.action.actions
        #pos_list=[[True,True],[True,False],[False,True],[False,False],no open]
        n= self.env.action_space.n
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
    
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        threshold: float = kwargs.get('threshold', 0)

        rand = random.random()
        if rand < threshold:
            if self.env.portfolio.occupy_rate > self.env.action.limit_order_margin_rate:
                return np.random.choice(self.action_idx_list[4])
            elif self.env.portfolio.long_pos > self.env.min_volume:
                if self.env.portfolio.short_pos > self.env.min_volume:
                    return np.random.choice(self.action_idx_list[0])
                else:
                    return np.random.choice(self.action_idx_list[1])
            else:
                if self.env.portfolio.short_pos > self.env.min_volume:
                    return np.random.choice(self.action_idx_list[2])
                else:
                    return np.random.choice(self.action_idx_list[3])
        else:
            return np.argmax(self.policy_network(np.expand_dims(state, 0)))

    def _apply_gradient_descent(self, memory: ReplayMemory, batch_size: int, learning_rate: float, discount_factor: float):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.Huber()

        transitions = memory.sample(batch_size)
        batch = DQNTransition(*zip(*transitions))

        state_batch = tf.convert_to_tensor(batch.state)
        action_batch = tf.convert_to_tensor(batch.action)
        reward_batch = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(batch.next_state)
        done_batch = tf.convert_to_tensor(batch.done)

        with tf.GradientTape() as tape:
            state_action_values = tf.math.reduce_sum(
                self.policy_network(state_batch) * tf.one_hot(action_batch, self.n_actions),
                axis=1
            )

            next_state_values = tf.where(
                done_batch,
                tf.zeros(batch_size),
                tf.math.reduce_max(self.target_network(next_state_batch), axis=1)
            )

            expected_state_action_values = reward_batch + (discount_factor * next_state_values)
            loss_value = loss(expected_state_action_values, state_action_values)

        variables = self.policy_network.trainable_variables
        gradients = tape.gradient(loss_value, variables)
        optimizer.apply_gradients(zip(gradients, variables))


    def train(self,
              n_steps: int = None,
              n_episodes: int = None,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        batch_size: int = kwargs.get('batch_size', 128)
        discount_factor: float = kwargs.get('discount_factor', 0.9999)
        learning_rate: float = kwargs.get('learning_rate', 0.0001)
        eps_start: float = kwargs.get('eps_start', 0.9)
        eps_end: float = kwargs.get('eps_end', 0.05)
        eps_decay_steps: int = kwargs.get('eps_decay_steps', 200)
        update_target_every: int = kwargs.get('update_target_every', 1000)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)
        render_interval: int = kwargs.get('render_interval', 50)  # in steps, None for episode end renderers only

        memory = ReplayMemory(memory_capacity, transition_type=DQNTransition)
        episode = 0
        total_steps_done = 0
        total_reward = 0
        stop_training = False

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))

        while episode < n_episodes and not stop_training:
            state = self.env.reset()
            done = False
            steps_done = 0

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-total_steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                memory.push(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                steps_done += 1
                total_steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory, batch_size, learning_rate, discount_factor)

                if n_steps and steps_done >= n_steps:
                    done = True

                if render_interval is not None and steps_done % render_interval == 0:
                    self.env.render(
                        episode=episode,
                        max_episodes=n_episodes,
                        max_steps=n_steps
                    )

                if steps_done % update_target_every == 0:
                    self.target_network = tf.keras.models.clone_model(self.policy_network)
                    self.target_network.trainable = False

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes - 1):
                self.save(save_path, episode=episode)

            if not render_interval or steps_done < n_steps:
                self.env.render(
                    episode=episode,
                    max_episodes=n_episodes,
                    max_steps=n_steps
                )  # renderers final state at episode end if not rendered earlier

            #self.env.save()

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward
