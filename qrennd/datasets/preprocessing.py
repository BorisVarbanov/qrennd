from itertools import product
from typing import List, Optional

import numpy as np
import xarray as xr
from scipy.special import erfcinv


def odd_parity(bits):
    return sum(bits) % 2


def norm_pdf(x: np.ndarray, means: List[float], dev: float):
    y = np.subtract(x, means) / dev
    prob = np.exp(-(y**2) / 2) / np.sqrt(2 * np.pi)
    return prob / dev


def get_syndromes(anc_meas: xr.DataArray) -> xr.DataArray:
    if anc_meas.meas_reset:
        return anc_meas

    shifted_meas = anc_meas.shift(qec_round=1, fill_value=0)
    syndromes = anc_meas ^ shifted_meas
    return syndromes


def get_defects(
    syndromes: xr.DataArray, frame: Optional[xr.DataArray] = None
) -> xr.DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    return defects


def get_final_defects(
    syndromes: xr.DataArray, proj_syndrome: xr.DataArray
) -> xr.DataArray:
    last_round = syndromes.qec_round.values[-1]
    anc_qubits = proj_syndrome.anc_qubit.values

    last_syndromes = syndromes.sel(anc_qubit=anc_qubits, qec_round=last_round)
    defects = last_syndromes ^ proj_syndrome
    return defects


def get_state_probs(
    outcomes: xr.DataArray, means: List[float], dev: float
) -> xr.DataArray:
    """
    get_state_probs Calculates the probabilities of the qubit
    being in a given state (0 or 1) given the soft measurement
    outcomes.

    Parameters
    ----------
    outcomes : xr.DataArray
        The soft measurement outcomes.
    means : List[float]
        The mean of the (projected 1D) Gaussian distributions for each state.
    dev : float
        The standard deviation of each Gaussian (assumed to be the same).

    Returns
    -------
    xr.DataArray
        The probabilities of the qubit being in each state given the outcomes.
    """
    probs_gen = (norm_pdf(outcomes, mean, dev) for mean in means)
    outcome_probs = xr.concat(probs_gen, dim="state")
    state_probs = outcome_probs / outcome_probs.sum(dim="state")
    return state_probs


def get_defect_probs(anc_probs: xr.DataArray) -> xr.DataArray:
    """
    get_defect_probs Calculates the probability of observing a defect, given
    the probabilities of the ancilla qubits being in a given state (0 or 1).

    Parameters
    ----------
    anc_probs : xr.DataArray
        The probabilities of each ancilla qubits being in a given state (0 or 1) over
        each round of the experiment.

    Returns
    -------
    xr.DataArray
        The probabilities of observing a defect at each round.
    """
    round_shift = 1 if anc_probs.meas_reset else 2

    shifted_probs = anc_probs.shift(qec_round=round_shift)
    prob_product = anc_probs.dot(shifted_probs, dims="state")
    defect_probs = 1 - prob_product.fillna(anc_probs.sel(state=0))
    return defect_probs


def to_measurements(
    dataset: xr.Dataset,
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
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
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
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
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


def to_defect_probs(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
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

    num_shots, _, _ = anc_meas.shape

    # Get Gaussian params
    means = np.array([-1, 1])
    midpoint = 0.5 * np.abs(means[0] - means[1])
    dev = midpoint / (np.sqrt(2) * erfcinv(2 * float(dataset.error_prob)))

    rng = np.random.default_rng(seed=int(dataset.seed))  # avoids TypeError

    samples = rng.normal(means, dev, size=(*anc_meas.shape, 2))
    anc_outcomes = xr.where(anc_meas, samples[..., 1], samples[..., 0])

    anc_probs = get_state_probs(anc_outcomes, means, dev)
    defect_probs = get_defect_probs(anc_probs)

    samples = rng.normal(means, dev, size=(*data_meas.shape, 2))
    data_outcomes = xr.where(data_meas, samples[..., 1], samples[..., 0])

    data_probs = get_state_probs(data_outcomes, means, dev)

    # final defects
    stab_qubits = proj_mat.anc_qubit.values
    data = np.zeros((num_shots, len(stab_qubits)))

    for ind, anc_qubit in enumerate(stab_qubits):
        # select measurement probabilities of the given detector
        sel_round = dataset.qec_round.values[-1]

        proj_vec = proj_mat.sel(anc_qubit=anc_qubit)

        proj_probs = data_probs.where(proj_vec, drop=True)
        anc_prob = anc_probs.sel(anc_qubit=anc_qubit, qec_round=sel_round)

        probs = np.concatenate((proj_probs.values, anc_prob.values[..., None]), axis=-1)

        num_qubits = proj_probs.data_qubit.size + 1
        products = product((0, 1), repeat=num_qubits)
        odd_products = filter(odd_parity, products)
        combinations = np.array(list(odd_products))

        comb_probs = np.where(combinations[:, None], probs[1], probs[0])
        defect_prob = np.sum(np.prod(comb_probs, axis=-1), axis=0)

        data[:, ind] = defect_prob

    final_defect_probs = xr.DataArray(
        data=data,
        dims=["shot", "anc_qubit"],
        coords=dict(shot=dataset.shot, anc_qubit=stab_qubits),
    )

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    inputs = dict(
        lstm_input=defect_probs.values,
        eval_input=final_defect_probs.values,
    )
    outputs = log_errors.values

    return inputs, outputs
