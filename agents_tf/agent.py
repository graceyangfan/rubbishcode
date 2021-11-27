import uuid
from abc import ABCMeta,abstractmethod
import numpy as np


class Identifiable(object, metaclass=ABCMeta):
    """Identifiable mixin for adding a unique `id` property to instances of a class.
    """

    @property
    def id(self) -> str:
        """Gets the identifier for the object.
        Returns
        -------
        str
           The identifier for the object.
        """
        if not hasattr(self, '_id'):
            self._id = str(uuid.uuid4())
        return self._id

    @id.setter
    def id(self, identifier: str) -> None:
        """Sets the identifier for the object
        Parameters
        ----------
        identifier : str
            The identifier to set for the object.
        """
        self._id = identifier


class Agent(Identifiable, metaclass=ABCMeta):

    @abstractmethod
    def restore(self, path: str, **kwargs):
        """Restore the agent from the file specified in `path`."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, path: str, **kwargs):
        """Save the agent to the directory specified in `path`."""
        raise NotImplementedError()

    @abstractmethod
    def get_action(self, state: np.ndarray, **kwargs) -> int:
        """Get an action for a specific state in the environment."""
        raise NotImplementedError()

    @abstractmethod
    def train(self,
              n_steps: int = None,
              n_episodes: int = 10000,
              save_every: int = None,
              save_path: str = None,
              callback: callable = None,
              **kwargs) -> float:
        """Train the agent in the environment and return the mean reward."""
        raise NotImplementedError()


