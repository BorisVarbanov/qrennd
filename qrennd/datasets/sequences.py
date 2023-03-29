from math import floor
from typing import Dict, Generator, List, Tuple, TypeVar

import numpy as np
from tensorflow.keras.utils import Sequence

RaggedSeq = TypeVar("RaggedSeq", bound="RaggedSequence")


class RaggedSequence(Sequence):
    def __init__(
        self,
        rec_inputs: List[np.ndarray],
        eval_inputs: List[np.ndarray],
        log_errors: List[np.ndarray],
        batch_size: int,
        predict_defects: bool = False,
    ) -> None:
        self.rec_inputs = rec_inputs
        self.eval_inputs = eval_inputs
        self.log_errors = log_errors

        self.num_groups = len(rec_inputs)
        self.group_size = rec_inputs[0].shape[0]

        self.batch_size = batch_size
        self.predict = predict_defects

    @classmethod
    def from_generator(
        cls: RaggedSeq,
        generator: Generator,
        batch_size: int,
        predict_defects: bool,
    ) -> "RaggedSequence":
        rec_inputs = []
        eval_inputs = []
        log_errors = []

        for rec_input, eval_input, log_error in generator:
            rec_inputs.append(rec_input)
            eval_inputs.append(eval_input)
            log_errors.append(log_error)

        return RaggedSequence(
            rec_inputs,
            eval_inputs,
            log_errors,
            batch_size,
            predict_defects,
        )

    def __len__(self) -> int:
        """
        __len__ Returns the number of batches per epoch

        Returns
        -------
        int
            Number of batches per epoch
        """
        return self.num_groups * floor(self.group_size / self.batch_size)

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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

        rec_inputs = self.rec_inputs[group][start_shot:end_shot]
        eval_inputs = self.eval_inputs[group][start_shot:end_shot]

        inputs = dict(rec_input=rec_inputs, eval_input=eval_inputs)

        log_errors = self.log_errors[group][start_shot:end_shot]

        if self.predict:
            rec_predictions = rec_inputs[:, 1:].reshape(self.batch_size, -1)
            predictions = np.concatenate((rec_predictions, eval_inputs), axis=1)
            outputs = dict(
                predictions=predictions, main_output=log_errors, aux_output=log_errors
            )
            return inputs, outputs

        outputs = dict(main_output=log_errors, aux_output=log_errors)
        return inputs, outputs
