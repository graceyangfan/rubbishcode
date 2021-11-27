import random
import numpy as np
import tensorflow as tf
from vnpy.trader.constant import (
    Status,
    Direction,
    Offset,
    Exchange
)
from typing import Callable
from zenquant.env.backtest_env import BacktestEnv 

class ParallelDQNModel:

    def __init__(self,
                 create_env: Callable[[], 'BacktestEnv'],
                 policy_network: tf.keras.Model = None):
        temp_env = create_env()
        self.temp_env = temp_env
        self.n_actions = temp_env.action_space.n
        self.observation_shape = temp_env.init_observation.shape

        self.policy_network = policy_network or self._build_policy_network()

        self.target_network = tf.keras.models.clone_model(self.policy_network)
        self.target_network.trainable = False
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
        agent_id: int = kwargs.get('agent_id', 'No_ID')
        episode: int = kwargs.get('episode', None)

        if episode:
            filename = "policy_network__" + agent_id + "__" + str(episode).zfill(3) + ".hdf5"
        else:
            filename = "policy_network__" + agent_id + ".hdf5"

        self.policy_network.save(path + filename)
    def get_valid_action_space(self):
        actions = self.temp_env.action.actions
        #pos_list=[[True,True],[True,False],[False,True],[False,False],no open]
        n= self.temp_env.action_space.n
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
            if self.temp_env.portfolio.occupy_rate > self.temp_env.action.limit_order_margin_rate:
                return np.random.choice(self.action_idx_list[4])
            elif self.temp_env.portfolio.long_pos > self.temp_env.min_volume:
                if self.temp_env.portfolio.short_pos > self.temp_env.min_volume:
                    return np.random.choice(self.action_idx_list[0])
                else:
                    return np.random.choice(self.action_idx_list[1])
            else:
                if self.temp_env.portfolio.short_pos > self.temp_env.min_volume:
                    return np.random.choice(self.action_idx_list[2])
                else:
                    return np.random.choice(self.action_idx_list[3])
        else:
            return np.argmax(self.policy_network(np.expand_dims(state, 0)))

    def update_networks(self, model: 'ParallelDQNModel'):
        self.policy_network.set_weights(model.policy_network.get_weights())
        self.target_network.set_weights(model.target_network.get_weights())

    def update_target_network(self):
        self.target_network.set_weights(self.policy_network.get_weights())