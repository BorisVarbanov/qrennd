from typing import Optional
from itertools import permutations

import numpy as np
from xarray import DataArray, Dataset
from scipy.special import erfcinv


def get_syndromes(anc_meas: DataArray) -> DataArray:
    if anc_meas.meas_reset:
        return anc_meas

    shifted_meas = anc_meas.shift(qec_round=1, fill_value=0)
    syndromes = anc_meas ^ shifted_meas
    return syndromes


def get_defects(syndromes: DataArray, frame: Optional[DataArray] = None) -> DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    return defects


def get_final_defects(syndromes: DataArray, proj_syndrome: DataArray) -> DataArray:
    last_round = syndromes.qec_round.values[-1]
    anc_qubits = proj_syndrome.anc_qubit.values

    last_syndromes = syndromes.sel(anc_qubit=anc_qubits, qec_round=last_round)
    defects = last_syndromes ^ proj_syndrome
    return defects


def to_measurements(
    dataset: Dataset,
):
    """
    Preprocess dataset to generate measurement inputs
    and the logical errors.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas: [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas: [qec_cycle, anc_qubit]
        - data_meas: [shot, data_qubit]
        - idea_data_meas: [data_qubit]
    """
    lstm_inputs = dataset.anc_meas
    eval_inputs = dataset.data_meas

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    inputs = dict(
        lstm_input=lstm_inputs.values.astype(bool),
        eval_input=eval_inputs.values.astype(bool),
    )
    outputs = log_errors.values.astype(bool)

    return inputs, outputs


def to_syndromes(
    dataset: Dataset,
    proj_mat: DataArray,
):
    """
    Preprocess dataset to generate syndrome inputs
    and the logical errors.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas: [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas: [qec_cycle, anc_qubit]
        - data_meas: [shot, data_qubit]
        - idea_data_meas: [data_qubit]
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    """
    anc_flips = dataset.anc_meas ^ dataset.ideal_anc_meas
    lstm_inputs = get_syndromes(anc_flips)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    eval_inputs = (data_flips @ proj_mat) % 2

    log_errors = data_flips.sum(dim="data_qubit") % 2

    inputs = dict(
        lstm_input=lstm_inputs.values.astype(bool),
        eval_input=eval_inputs.values.astype(bool),
    )
    outputs = log_errors.values.astype(bool)

    return inputs, outputs


def to_defects(
    dataset: Dataset,
    proj_mat: DataArray,
):
    """
    Preprocess dataset to generate defect inputs
    and the logical errors.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas: [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas: [qec_cycle, anc_qubit]
        - data_meas: [shot, data_qubit]
        - idea_data_meas: [data_qubit]
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    """
    anc_flips = dataset.anc_meas ^ dataset.ideal_anc_meas
    syndromes = get_syndromes(anc_flips)
    lstm_inputs = get_defects(syndromes)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    proj_syndrome = (data_flips @ proj_mat) % 2
    eval_inputs = get_final_defects(syndromes, proj_syndrome)

    log_errors = data_flips.sum(dim="data_qubit") % 2

    inputs = dict(
        lstm_input=lstm_inputs.values.astype(bool),
        eval_input=eval_inputs.values.astype(bool),
    )
    outputs = log_errors.values.astype(bool)

    return inputs, outputs


def to_prob_defects(
    dataset: Dataset,
    proj_mat: DataArray,
):
    """
    Preprocess dataset to generate the probability of defect
    based on the soft outcomes and the logical errors.
    Assumes ideal measurements are 0s.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas: [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas: [qec_cycle, anc_qubit]
        - data_meas: [shot, data_qubit]
        - idea_data_meas: [data_qubit]
        - prob_error: float
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    """

    assert dataset.meas_reset
    anc_meas = dataset.anc_meas.transpose(..., "qec_round", "anc_qubit")
    data_meas = dataset.data_meas.transpose(..., "data_qubit")
    n_shots, n_rounds, n_anc = anc_meas.shape

    # Get Gaussian params
    meas_error = 1 / (2 * np.sqrt(2) * erfcinv(2 * dataset.prob_error))
    rng = np.random.default_rng(seed=int(dataset.seed.values))  # avoids TypeError

    anc_gauss = rng.normal(scale=meas_error, size=anc_meas.shape)
    anc_gauss = DataArray(data=anc_gauss, coords=anc_meas.coords)
    soft_outcomes = anc_gauss + anc_meas
    prob_0 = np.exp(-((soft_outcomes) ** 2) / (2 * meas_error**2))
    prob_1 = np.exp(-((soft_outcomes - 1) ** 2) / (2 * meas_error**2))
    total = prob_0 + prob_1
    anc_prob_0 = prob_0 / total
    anc_prob_1 = prob_1 / total

    data_gauss = rng.normal(scale=meas_error, size=data_meas.shape)
    data_gauss = DataArray(data=data_gauss, coords=data_meas.coords)
    soft_outcomes = data_gauss + data_meas
    prob_0 = np.exp(-((soft_outcomes) ** 2) / (2 * meas_error**2))
    prob_1 = np.exp(-((soft_outcomes - 1) ** 2) / (2 * meas_error**2))
    total = prob_0 + prob_1
    data_prob_0 = prob_0 / total
    data_prob_1 = prob_1 / total

    # defects
    prob_defects = anc_prob_1 * anc_prob_0.shift(
        qec_round=-1, fill_value=1
    ) + anc_prob_0 * anc_prob_1.shift(qec_round=-1, fill_value=0)

    # final defects
    prob_final_defects = np.zeros((n_shots, n_anc))
    for anc_idx, ancilla in enumerate(proj_mat.anc_qubit):
        # select measurement probabilities of the given detector
        detector = proj_mat.sel(anc_qubit=ancilla)
        probs_0 = [
            data_prob_0.sel(data_qubit=q)
            for q in detector.data_qubit
            if detector.sel(data_qubit=q)
        ]
        probs_1 = [
            data_prob_1.sel(data_qubit=q)
            for q in detector.data_qubit
            if detector.sel(data_qubit=q)
        ]
        probs_0.append(anc_prob_0.sel(anc_qubit=ancilla).isel(qec_round=-1))
        probs_1.append(anc_prob_1.sel(anc_qubit=ancilla).isel(qec_round=-1))

        # generate combinations
        combinations = set()
        n_qubits = len(probs_0)
        for odd in range(1, n_qubits + 1, 2):
            tmp = [k < odd for k in range(n_qubits)]  # odd=3, n_qubits=5: [1,1,1,0,0]
            combinations = combinations.union(set(permutations(tmp)))

        # calculate probabililities
        defect_prob = 0
        for combination in combinations:
            combination = np.array(combination)[:, np.newaxis]
            defect_prob += np.product(
                np.where(np.repeat(combination, n_shots, axis=1), probs_1, probs_0),
                axis=0,
            )

        prob_final_defects[:, anc_idx] = defect_prob

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    inputs = dict(
        lstm_input=prob_defects.values,
        eval_input=prob_final_defects,
    )
    outputs = log_errors.values.astype(bool)

    return inputs, outputs
