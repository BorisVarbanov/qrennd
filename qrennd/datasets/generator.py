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
        rot_basis: bool,
        batch_size: int,
        lstm_input: str,
        eval_input: str,
        folder_format_name: str,
        proj_matrix: Optional[xr.DataArray] = None,
    ) -> None:
        num_states = len(states)
        num_rounds = len(rounds)

        self.num_groups = num_rounds
        self.group_size = shots * num_states
        self.batch_size = batch_size

        self._lstm_inputs = []
        self._eval_inputs = []
        self._outputs = []
        self.load_datasets(
            dirpath=dirpath,
            shots=shots,
            states=states,
            rounds=rounds,
            rot_basis=rot_basis,
            lstm_input=lstm_input,
            eval_input=eval_input,
            folder_format_name=folder_format_name,
            proj_matrix=proj_matrix,
        )

    def load_datasets(
        self,
        dirpath: str,
        shots: int,
        states: List[int],
        rounds: List[int],
        rot_basis: bool,
        lstm_input: str,
        eval_input: str,
        folder_format_name: str,
        proj_matrix: Optional[xr.DataArray] = None,
    ) -> xr.Dataset:
        rot_basis = "X" if rot_basis else "Z"
        for num_rounds in rounds:
            _datasets = []
            for state in states:
                experiment = folder_format_name.format(
                    state=state,
                    shots=shots,
                    num_rounds=num_rounds,
                    rot_basis=rot_basis,
                )
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
        group = index % self.num_groups

        batch_ind = index // self.num_groups
        start_shot = batch_ind * self.batch_size
        end_shot = start_shot + self.batch_size

        inputs = dict(
            lstm_input=self._lstm_inputs[group][start_shot:end_shot],
            eval_input=self._eval_inputs[group][start_shot:end_shot],
        )
        outputs = self._outputs[group][start_shot:end_shot]
        return inputs, outputs


class DataGeneratorGoogle(Sequence):
    def __init__(
        self,
        dirpath: Path,
        shots: int,
        rounds: List[int],
        rot_basis: bool,
        batch_size: int,
        lstm_input: str,
        eval_input: str,
        folder_format_name: str,
        proj_matrix: Optional[xr.DataArray] = None,
    ) -> None:
        num_rounds = len(rounds)

        self.num_groups = num_rounds
        self.group_size = shots
        self.batch_size = batch_size

        self._lstm_inputs = []
        self._eval_inputs = []
        self._outputs = []
        self.load_datasets(
            dirpath=dirpath,
            shots=shots,
            rounds=rounds,
            rot_basis=rot_basis,
            lstm_input=lstm_input,
            eval_input=eval_input,
            folder_format_name=folder_format_name,
            proj_matrix=proj_matrix,
        )

    def load_datasets(
        self,
        dirpath: str,
        shots: int,
        rounds: List[int],
        rot_basis: bool,
        lstm_input: str,
        eval_input: str,
        folder_format_name: str,
        proj_matrix: Optional[xr.DataArray] = None,
    ) -> xr.Dataset:
        rot_basis = "X" if rot_basis else "Z"
        for num_rounds in rounds:
            experiment = folder_format_name.format(
                rot_basis=rot_basis, shots=shots, num_rounds=num_rounds
            )
            dataset = xr.open_dataset(dirpath / experiment / "measurements.nc")

            anc_meas = dataset.anc_meas.transpose("shot", "qec_round", "anc_qubit")
            data_meas = dataset.data_meas.transpose("shot", "data_qubit")
            sweeps = dataset.sweeps.transpose("shot", "data_qubit")
            sweeps = sweeps ^ np.array(
                [False, True, False, True, False, True, False, True, False]
            )

            syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)

            if lstm_input == "measurements":
                self._lstm_inputs.append(anc_meas.values)
            elif lstm_input == "syndromes":
                self._lstm_inputs.append(syndromes.values)
            elif lstm_input == "defects":
                # get frame
                proj_matrix = proj_matrix.sel(anc_qubit=["Z1", "Z2", "Z3", "Z4"])
                frame_z = (sweeps @ proj_matrix) % 2
                frame_x = anc_meas.sel(qec_round=1, anc_qubit=["X1", "X2", "X3", "X4"])
                frame = xr.concat([frame_x, frame_z], dim="anc_qubit")
                # get defects
                defects = get_defects(syndromes, frame=frame)
                # apply corrections due to Google's modifications to the QEC circuit
                defects = defects ^ np.array(
                    [True, False, False, True, False, True, True, False]
                )
                initial_def = defects.sel(qec_round=1)
                initial_def = initial_def ^ np.array(
                    [[True, False, False, False, False, True, False, False]]
                )
                other = defects.sel(qec_round=np.arange(2, len(dataset.qec_round) + 1))
                defects = xr.concat([initial_def, other], dim="qec_round")
                defects = defects.transpose("shot", "qec_round", "anc_qubit")
                self._lstm_inputs.append(defects.values)
            else:
                raise TypeError(
                    "'data_input' must be 'defects', 'syndrmes', or 'measurements'"
                )

            if eval_input == "measurements":
                self._eval_inputs.append(data_meas.values)
            elif eval_input == "defects":
                proj_matrix = proj_matrix.sel(anc_qubit=["Z4", "Z3", "Z2", "Z1"])
                proj_syndrome = (data_meas @ proj_matrix) % 2
                final_defects = get_final_defects(syndromes, proj_syndrome)
                # apply corrections due to Google's modifications to the QEC circuit
                final_defects = final_defects ^ np.array([False, True, False, False])
                self._eval_inputs.append(final_defects.values)
            else:
                raise TypeError(
                    "'data_final_input' must be 'defects' or 'measurements'"
                )

            log_meas = (
                data_meas.sel(data_qubit=["D9", "D8", "D7"]).sum(dim="data_qubit") % 2
            )
            log_state = sweeps.sum(dim="data_qubit") % 2
            log_errors = log_meas ^ log_state
            log_errors = log_errors.values

            self._outputs.append(log_errors)

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
        group = index % self.num_groups

        batch_ind = index // self.num_groups
        start_shot = batch_ind * self.batch_size
        end_shot = start_shot + self.batch_size

        inputs = dict(
            lstm_input=self._lstm_inputs[group][start_shot:end_shot],
            eval_input=self._eval_inputs[group][start_shot:end_shot],
        )
        outputs = self._outputs[group][start_shot:end_shot]
        return inputs, outputs
