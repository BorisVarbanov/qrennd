from math import floor
from typing import Dict, Generator, List, Tuple, TypeVar

from numpy import ndarray
from tensorflow.keras.utils import Sequence

RaggedSeq = TypeVar("RaggedSeq", bound="RaggedSequence")


class RaggedSequence(Sequence):
    def __init__(
        self,
        lstm_inputs: List[ndarray],
        eval_inputs: List[ndarray],
        outputs: List[ndarray],
        batch_size: int,
    ) -> None:
        self.lstm_inputs = lstm_inputs
        self.eval_inputs = eval_inputs
        self.outputs = outputs

        self.num_groups = len(lstm_inputs)
        self.group_size = lstm_inputs[0].shape[0]

        self.batch_size = batch_size

    @classmethod
    def from_generator(
        cls: RaggedSeq,
        generator: Generator,
        batch_size: int,
    ) -> "RaggedSequence":
        lstm_inputs = []
        eval_inputs = []
        outputs = []

        for inputs, output in generator:
            lstm_inputs.append(inputs["lstm_input"])
            eval_inputs.append(inputs["eval_input"])
            outputs.append(output)

        return RaggedSequence(lstm_inputs, eval_inputs, outputs, batch_size)

    def __len__(self) -> int:
        """
        __len__ Returns the number of batches per epoch

        Returns
        -------
        int
            Number of batches per epoch
        """
        return self.num_groups * floor(self.group_size / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[Dict[str, ndarray], ndarray]:
        """
        __getitem__ Returns a single batch of the dataset

        Returns
        -------
        Tuple[Dict[str, NDArray], NDArray]
            A tuple of a dictionary with the input data, consisting of the defects and
            final defects, together with the output label.
        """
        group = index % self.num_groups

        batch_ind = index // self.num_groups
        start_shot = batch_ind * self.batch_size
        end_shot = start_shot + self.batch_size

        inputs = dict(
            lstm_input=self.lstm_inputs[group][start_shot:end_shot],
            eval_input=self.eval_inputs[group][start_shot:end_shot],
        )

        outputs = self.outputs[group][start_shot:end_shot]
        return inputs, outputs
