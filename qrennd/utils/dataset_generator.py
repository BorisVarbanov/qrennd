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


class DataGenerator(Sequence):
    def __init__(
        self,
        folder: Path,
        num_shots: int,
        states: List[int],
        qec_rounds: List[int],
        batch_size: int,
        data_input: str,
        data_final_input: str,
        proj_matrix: Optional[xr.DataArray],
    ):
        num_states = len(states)
        num_rounds = len(qec_rounds)

        self.num_groups = num_rounds
        self.group_size = num_shots * num_states

        self.batch_size = batch_size

        self.data_input = data_input
        self.data_final_input = data_final_input
        self.proj_matrix = proj_matrix

        self._inputs = []
        self._aux_inputs = []
        self._outputs = []
        self.load_datasets(
            folder,
            states,
            num_shots,
            qec_rounds,
            data_input,
            data_final_input,
            proj_matrix,
        )

    def load_datasets(
        self,
        folder: str,
        states: List[int],
        num_shots: int,
        qec_rounds: List[int],
        data_input: str,
        data_final_input: str,
        proj_matrix: Optional[xr.DataArray],
    ) -> xr.Dataset:
        for num_rounds in qec_rounds:
            _datasets = []
            for state in states:
                experiment = f"surf-code_d3_bZ_s{state}_n{num_shots}_r{num_rounds}"
                dataset = xr.open_dataset(folder / experiment / "measurements.nc")
                _datasets.append(dataset)

            dataset = xr.concat(_datasets, dim="log_state")
            dataset = dataset.stack(run=["log_state", "shot"])

            meas_reset = dataset.meas_reset

            anc_meas = dataset.anc_meas.transpose("run", "qec_round", "anc_qubit")
            syndromes = get_syndromes(anc_meas, meas_reset)
            defects = get_defects(syndromes)
            if input_type == "measurements":
                self._inputs.append(anc_meas.values)
            elif input_type == "syndromes":
                self._inputs.append(syndromes.values)
            elif input_type == "defects":
                self._inputs.append(defects.values)
            else:
                raise TypeError(
                    "'input_type' must be 'defects', 'syndrmes', or 'measurements'"
                )

            data_meas = dataset.data_meas.transpose("run", "data_qubit")
            final_defects = get_final_defects(syndromes, proj_matrix)
            if final_input_type == "measurements":
                self._aux_inputs.append(data_meas.values)
            elif final_input_type == "defects":
                self._aux_inputs.append(final_defects.values)
            else:
                raise TypeError(
                    "'final_input_type' must be 'defects' or 'measurements'"
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
            defects=self._inputs[group_ind][start_ind:end_ind],
            final_defects=self._aux_inputs[group_ind][start_ind:end_ind],
        )
        outputs = self._outputs[group_ind][start_ind:end_ind]
        return inputs, outputs
