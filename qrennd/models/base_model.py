"""Abstrace base model class."""
from abc import ABC, abstractmethod

from ..utils import Config


class BaseModel(ABC):
    """Abstract base neura  l network model."""

    def __init__(self, config: Config) -> None:
        """
        __init__ Initializes the model.

        Parameters
        ----------
        config : Config
            The Config object containing the model hyperparameters.
        """
        self.config = config

    @abstractmethod
    def save(self) -> None:
        """Save the model."""
        pass

    @abstractmethod
    def build(self) -> None:
        """Build the model."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def eval(self) -> None:
        """Evaluate the model."""
        pass
