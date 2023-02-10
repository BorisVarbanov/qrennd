# %%
# Module import
from math import floor
from pathlib import Path
from typing import List, Optional

import xarray as xr
from tensorflow.keras.utils import Sequence

from qrennd.utils.data_processing import (
    get_defects,
    get_syndromes,
    get_final_defects,
)
from qrennd import Config

def get_seq_input():
    dataset = xr.concat(_datasets, dim="log_state")
    dataset = dataset.stack(run=["log_state", "shot"])

    anc_meas = dataset.anc_meas.transpose("run", "qec_round", "anc_qubit")
    syndromes = get_syndromes(anc_meas, dataset.meas_reset)

    if data_input == "measurements":
        self._inputs.append(anc_meas.values)
    elif data_input == "syndromes":
        self._inputs.append(syndromes.values)
    elif data_input == "defects":
        defects = get_defects(syndromes)
        self._inputs.append(defects.values)
    else:
        raise TypeError(
            "'data_input' must be 'defects', 'syndrmes', or 'measurements'"
        )



class DataGenerator(Sequence):
    def __init__(
        self,
        seq_inputs,
        outputs,
    ):
        num_states = len(states)
        num_rounds = len(qec_rounds)

        self.num_groups = num_rounds
        self.group_size = num_shots * num_states

        self.batch_size = batch_size

        self.data_input = data_input
        self.data_final_input = data_final_input

        self.seq_inputs = []
        self.aux_input = []

        self.output = []

    @classmethod
    def load_datasets(
        cls,
        path: str,
        config: Config,
        proj_matrix: Optional[xr.DataArray],
    ) -> xr.Dataset:

        rounds = config.dataset["rounds"]
        states = config.dataset["states"]
        shots = config.dataset["shots"]

        for num_rounds in rounds:
            _datasets = []
            for state in states:
                experiment = f"surf-code_d3_bZ_s{state}_n{shots}_r{num_rounds}"
                dataset = xr.open_dataset(path / experiment / "measurements.nc")
                _datasets.append(dataset)


            data_meas = dataset.data_meas.transpose("run", "data_qubit")
            if data_final_input == "measurements":
                self._aux_inputs.append(data_meas.values)
            elif data_final_input == "defects":
                final_syndromes = (data_meas @ proj_matrix) % 2
                final_defects = get_final_defects(syndromes, final_syndromes)
                self._aux_inputs.append(final_defects.values)
            else:
                raise TypeError(
                    "'data_final_input' must be 'defects' or 'measurements'"
                )

            log_meas = data_meas.sum(dim="data_qubit") % 2
            log_errors = log_meas ^ dataset.log_state
            self._outputs.append(log_errors.values)

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
        batch_ind = index // self.num_groups

        start_ind = batch_ind * self.batch_size
        end_ind = start_ind + self.batch_size

        inputs = dict(
            seq_inputs=self.seq_inputs[group_ind][start_ind:end_ind],
            aux_input=self.aux_input[group_ind][start_ind:end_ind],
        )
        output = self.output[group_ind][start_ind:end_ind]
        return inputs, output
