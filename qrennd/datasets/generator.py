# %%
# Module import
from math import floor
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr
from tensorflow.keras.utils import Sequence

from .preprocess import get_defects, get_final_defects, get_syndromes


class DataGenerator(Sequence):
    def __init__(
        self,
        dirpath: Path,
        shots: int,
        states: List[int],
        rounds: List[int],
        batch_size: int,
        lstm_input: str,
        eval_input: str,
        shuffle: bool = True,
        seed: Optional[int] = None,
        proj_matrix: Optional[xr.DataArray] = None,
    ) -> None:
        num_states = len(states)
        num_rounds = len(rounds)

        self.num_groups = num_rounds
        self.group_size = shots * num_states
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

        self._groups = np.arange(self.num_groups)

        self._lstm_inputs = []
        self._eval_inputs = []
        self._outputs = []
        self.load_datasets(
            dirpath, shots, states, rounds, lstm_input, eval_input, proj_matrix
        )

    def load_datasets(
        self,
        dirpath: str,
        shots: int,
        states: List[int],
        rounds: List[int],
        lstm_input: str,
        eval_input: str,
        proj_matrix: Optional[xr.DataArray],
    ) -> xr.Dataset:
        for num_rounds in rounds:
            _datasets = []
            for state in states:
                experiment = f"surf-code_d3_bZ_s{state}_n{shots}_r{num_rounds}"
                dataset = xr.open_dataset(dirpath / experiment / "measurements.nc")
                _datasets.append(dataset)

            dataset = xr.concat(_datasets, dim="log_state")
            dataset = dataset.stack(run=["log_state", "shot"])

            meas_reset = dataset.meas_reset

            anc_meas = dataset.anc_meas.transpose("run", "qec_round", "anc_qubit")
            syndromes = get_syndromes(anc_meas, meas_reset)

            if lstm_input == "measurements":
                self._lstm_inputs.append(anc_meas.values)
            elif lstm_input == "syndromes":
                self._lstm_inputs.append(syndromes.values)
            elif lstm_input == "defects":
                defects = get_defects(syndromes)
                self._lstm_inputs.append(defects.values)
            else:
                raise TypeError(
                    "'data_input' must be 'defects', 'syndrmes', or 'measurements'"
                )

            data_meas = dataset.data_meas.transpose("run", "data_qubit")
            if eval_input == "measurements":
                self._eval_inputs.append(data_meas.values)
            elif eval_input == "defects":
                final_syndromes = (data_meas @ proj_matrix) % 2
                final_defects = get_final_defects(syndromes, final_syndromes)
                self._eval_inputs.append(final_defects.values)
            else:
                raise TypeError(
                    "'data_final_input' must be 'defects' or 'measurements'"
                )

            log_meas = data_meas.sum(dim="data_qubit") % 2
            log_errors = log_meas ^ dataset.log_state
            self._outputs.append(log_errors.values)

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self._groups)

            inds = np.arange(self.group_size)
            for group in range(self.num_groups):
                self.rng.shuffle(inds)

                self._lstm_inputs[group] = self._lstm_inputs[group][inds]
                self._eval_inputs[group] = self._eval_inputs[group][inds]
                self._outputs[group] = self._outputs[group][inds]

    def __len__(self) -> int:
        """
        __len__ Returns the number of batches per epoch

        Returns
        -------
        int
            Number of batches per epoch
        """
        return self.num_groups * floor(self.group_size / self.batch_size)

    def __getitem__(self, index: int):
        """
        __getitem__ Returns a single batch of the dataset

        Returns
        -------
        Tuple[Dict[str, NDArray], NDArray]
            A tuple of a dictionary with the input data, consisting of the defects and
            final defects, together with the output label.
        """
        group_ind = index % self.num_groups
        group = self._groups[group_ind]

        batch_ind = index // self.num_groups
        start_shot = batch_ind * self.batch_size
        end_shot = start_shot + self.batch_size

        inputs = dict(
            lstm_input=self._lstm_inputs[group][start_shot:end_shot],
            eval_input=self._eval_inputs[group][start_shot:end_shot],
        )
        outputs = self._outputs[group][start_shot:end_shot]
        return inputs, outputs
