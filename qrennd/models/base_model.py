"""Abstrace base model class."""
from abc import ABC, abstractmethod
from typing import Any, TypeVar

T = TypeVar("T", bound="BaseModel")


class BaseModel(ABC):
    """Abstract base neural network model."""

    def __init__(self: T, config: Any) -> None:
        """
        __init__ Initialize the model.

        Parameters
        ----------
        self : T
            BaseModel class
        config : Any
            The model configuration
        """
        self.config = config

    @abstractmethod
    def save(self: T) -> None:
        """Save the model."""
        pass

    @abstractmethod
    def build(self: T) -> None:
        """Build the model."""
        pass

    @abstractmethod
    def train(self: T) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def eval(self: T) -> None:
        """Evaluate the model."""
        pass
