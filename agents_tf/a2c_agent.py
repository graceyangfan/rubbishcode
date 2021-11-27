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

A2CTransition = namedtuple('A2CTransition', ['state', 'action', 'reward', 'done', 'value'])


class A2CAgent(Agent):

    def __init__(self,
                 env: 'BacktestEnv',
                 shared_network: tf.keras.Model = None,
                 actor_network: tf.keras.Model = None,
                 critic_network: tf.keras.Model = None):
        self.env = env
        self.n_actions = env.action_space.n
        self.observation_shape = env.init_observation.shape

        self.shared_network = shared_network or self._build_shared_network()
        self.actor_network = actor_network or self._build_actor_network()
        self.critic_network = critic_network or self._build_critic_network()

        self.env.agent_id = self.id
        self.action_idx_list = self.get_valid_action_space()
    def _build_shared_network(self):
        network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.observation_shape),
            tf.keras.layers.Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten()
        ])

        return network

    def _build_actor_network(self):
        actor_head = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(self.n_actions, activation='relu')
        ])

        return tf.keras.Sequential([self.shared_network, actor_head])

    def _build_critic_network(self):
        critic_head = tf.keras.Sequential([
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1, activation='relu')
        ])

        return tf.keras.Sequential([self.shared_network, critic_head])

    def restore(self, path: str, **kwargs):
        actor_filename: str = kwargs.get('actor_filename', None)
        critic_filename: str = kwargs.get('critic_filename', None)

        if not actor_filename or not critic_filename:
            raise ValueError(
                'The `restore` method requires a directory `path`, a `critic_filename`, and an `actor_filename`.')

        self.actor_network = tf.keras.models.load_model(path + actor_filename)
        self.critic_network = tf.keras.models.load_model(path + critic_filename)

    def save(self, path: str, **kwargs):
        episode: int = kwargs.get('episode', None)

        if episode:
            suffix = self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
            actor_filename = "actor_network__" + suffix
            critic_filename = "critic_network__" + suffix
        else:
            actor_filename = "actor_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"
            critic_filename = "critic_network__" + self.id[:7] + "__" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".hdf5"

        self.actor_network.save(path + actor_filename)
        self.critic_network.save(path + critic_filename)
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
            logits = self.actor_network(state[None, :], training=False)
            return tf.squeeze(tf.squeeze(tf.random.categorical(logits, 1), axis=-1), axis=-1)

    def _apply_gradient_descent(self,
                                memory: ReplayMemory,
                                batch_size: int,
                                learning_rate: float,
                                discount_factor: float,
                                entropy_c: float,):
        huber_loss = tf.keras.losses.Huber()
        wsce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        transitions = memory.tail(batch_size)
        batch = A2CTransition(*zip(*transitions))

        states = tf.convert_to_tensor(batch.state)
        actions = tf.convert_to_tensor(batch.action)
        rewards = tf.convert_to_tensor(batch.reward, dtype=tf.float32)
        dones = tf.convert_to_tensor(batch.done)
        values = tf.convert_to_tensor(batch.value)

        returns = []
        exp_weighted_return = 0

        for reward, done in zip(rewards[::-1], dones[::-1]):
            exp_weighted_return = reward + discount_factor * exp_weighted_return * (1 - int(done))
            returns += [exp_weighted_return]

        returns = returns[::-1]

        with tf.GradientTape() as tape:
            state_values = self.critic_network(states)
            critic_loss_value = huber_loss(returns, state_values)

        gradients = tape.gradient(critic_loss_value, self.critic_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.critic_network.trainable_variables))

        with tf.GradientTape() as tape:
            returns = tf.reshape(returns, [batch_size, 1])
            advantages = returns - values

            actions = tf.cast(actions, tf.int32)
            logits = self.actor_network(states)
            policy_loss_value = wsce_loss(actions, logits, sample_weight=advantages)

            probs = tf.nn.softmax(logits)
            entropy_loss_value = tf.keras.losses.categorical_crossentropy(probs, probs)
            policy_total_loss_value = policy_loss_value - entropy_c * entropy_loss_value

        gradients = tape.gradient(policy_total_loss_value,
                                  self.actor_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.actor_network.trainable_variables))

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
        entropy_c: int = kwargs.get('entropy_c', 0.0001)
        memory_capacity: int = kwargs.get('memory_capacity', 1000)

        memory = ReplayMemory(memory_capacity, transition_type=A2CTransition)
        episode = 0
        steps_done = 0
        total_reward = 0
        stop_training = False

        if n_steps and not n_episodes:
            n_episodes = np.iinfo(np.int32).max

        print('====      AGENT ID: {}      ===='.format(self.id))

        while episode < n_episodes and not stop_training:
            state = self.env.reset()
            done = False

            print('====      EPISODE ID ({}/{}): {}      ===='.format(episode + 1,
                                                                      n_episodes,
                                                                      self.env.episode_id))

            while not done:
                threshold = eps_end + (eps_start - eps_end) * np.exp(-steps_done / eps_decay_steps)
                action = self.get_action(state, threshold=threshold)
                next_state, reward, done, _ = self.env.step(action)

                value = self.critic_network(state[None, :], training=False)
                value = tf.squeeze(value, axis=-1)

                memory.push(state, action, reward, done, value)

                state = next_state
                total_reward += reward
                steps_done += 1

                if len(memory) < batch_size:
                    continue

                self._apply_gradient_descent(memory,
                                             batch_size,
                                             learning_rate,
                                             discount_factor,
                                             entropy_c)

                if n_steps and steps_done >= n_steps:
                    done = True
                    stop_training = True

            is_checkpoint = save_every and episode % save_every == 0

            if save_path and (is_checkpoint or episode == n_episodes):
                self.save(save_path, episode=episode)

            episode += 1

        mean_reward = total_reward / steps_done

        return mean_reward