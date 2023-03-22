from itertools import permutations
from typing import Optional

import numpy as np
from scipy.special import erfcinv
from xarray import DataArray, Dataset


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


def norm_pdf(x, loc, scale):
    y = np.subtract(x, loc) / scale
    prob = np.exp(-(y**2) / 2) / np.sqrt(2 * np.pi)
    return prob / scale


def state_probs(measurements, means, dev, rng):
    samples = rng.normal(means, dev, size=(*measurements.shape, 2))
    soft_outcomes = np.where(measurements, samples[..., 1], samples[..., 0])

    outcome_probs = norm_pdf(soft_outcomes[..., None], means, dev)
    outcome_probs = np.moveaxis(outcome_probs, -1, 0)

    state_probs = outcome_probs / np.sum(outcome_probs, axis=0)
    return state_probs


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

    anc_meas = dataset.anc_meas.transpose(..., "qec_round", "anc_qubit")
    data_meas = dataset.data_meas.transpose(..., "data_qubit")

    num_shots, _, num_anc = anc_meas.shape
    round_shift = -1 if dataset.meas_reset else -2

    # Get Gaussian params
    means = np.array([-1, 1])
    midpoint = 0.5 * np.abs(means[0] - means[1])
    dev = midpoint / (np.sqrt(2) * erfcinv(2 * float(dataset.error_prob)))

    rng = np.random.default_rng(seed=int(dataset.seed))  # avoids TypeError

    data = get_state_probs(anc_meas.values, means, dev, rng)
    anc_probs = DataArray(
        data=data,
        dims=("state", *anc_meas.dims),
        coords=dict(state=[0, 1], **anc_meas.coords),
    )

    data = get_state_probs(data_meas.values, means, dev, rng)
    data_probs = DataArray(
        data=data,
        dims=("state", *data_meas.dims),
        coords=dict(state=[0, 1], **data_meas.coords),
    )

    # defects
    prob_defects = anc_probs[0] * anc_probs[1].shift(
        qec_round=round_shift, fill_value=0
    )
    prob_defects += anc_probs[1] * anc_probs[0].shift(
        qec_round=round_shift, fill_value=1
    )

    # final defects
    prob_final_defects = np.zeros((num_shots, num_anc))
    for anc_idx, ancilla in enumerate(proj_mat.anc_qubit):
        # select measurement probabilities of the given detector
        detector = proj_mat.sel(anc_qubit=ancilla)
        probs_0 = [
            data_probs[0].sel(data_qubit=q)
            for q in detector.data_qubit
            if detector.sel(data_qubit=q)
        ]
        probs_1 = [
            data_probs[1].sel(data_qubit=q)
            for q in detector.data_qubit
            if detector.sel(data_qubit=q)
        ]
        probs_0.append(anc_probs[0].sel(anc_qubit=ancilla).isel(qec_round=-1))
        probs_1.append(anc_probs[1].sel(anc_qubit=ancilla).isel(qec_round=-1))

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
                np.where(np.repeat(combination, num_shots, axis=1), probs_1, probs_0),
                axis=0,
            )

        prob_final_defects[:, anc_idx] = defect_prob

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    return prob_defects, prob_final_defects, log_errors
