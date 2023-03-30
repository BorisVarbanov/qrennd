from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.special import erfcinv


def dev_from_error(means: Tuple[float, float], error_prob: float) -> float:
    ground_mean, exc_mean = means
    midpoint = 0.5 * np.abs(ground_mean - exc_mean)
    dev = midpoint / (np.sqrt(2) * erfcinv(2 * error_prob))
    return dev


def odd_parity(bits: Sequence[int]) -> int:
    return sum(bits) % 2


def norm_pdf(x: np.ndarray, means: Tuple[float, float], dev: float) -> np.ndarray:
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
    outcomes: xr.DataArray, means: Tuple[float, float], dev: float
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


def get_final_defect_probs(
    anc_probs: xr.DataArray,
    data_probs: xr.DataArray,
    proj_mat: xr.DataArray,
) -> xr.DataArray:
    """
    get_final_defect_probs Calculates the final
    defect probabilities.

    Parameters
    ----------
    anc_probs : xr.DataArray
        The probabilities of each ancilla qubits being in a given state (0 or 1)
        over each round of the experiment.
    data_probs : xr.DataArray
        The probabilities of each data qubits being in a given state (0 or 1)
        at the end of the experiment.
    proj_mat : xr.DataArray
        The projection matrix mapping the data qubits to the qubits that stabilize them (for
        the basis that the experiment is done in).

    Returns
    -------
    xr.DataArray
        The final defect probabilities.
    """
    round_shift = 1 if anc_probs.meas_reset else 2
    comp_rounds = anc_probs.qec_round[-round_shift:]
    comp_probs = anc_probs.sel(qec_round=comp_rounds)

    # Relabel to detector for concatenation later on
    # This was the smartest way I cound figure how to do this in xarray
    # Other option was to just
    comp_probs = comp_probs.rename(qec_round="detector")
    _data_probs = data_probs.rename(data_qubit="detector")
    _proj_mat = proj_mat.rename(data_qubit="detector")

    stab_qubits = proj_mat.anc_qubit.values
    shots = data_probs.shot.values

    data = np.zeros((shots.size, stab_qubits.size))
    final_defect_probs = xr.DataArray(
        data,
        dims=["shot", "anc_qubit"],
        coords=dict(shot=shots, anc_qubit=stab_qubits),
    )

    for ind, stab_qubit in enumerate(stab_qubits):
        proj_vec = _proj_mat.sel(anc_qubit=stab_qubit)
        data_det_probs = _data_probs.where(proj_vec, drop=True)

        anc_det_probs = comp_probs.sel(anc_qubit=stab_qubit)

        probs = xr.concat((data_det_probs, anc_det_probs), dim="detector")

        products = product((0, 1), repeat=probs.detector.size)
        odd_products = filter(odd_parity, products)
        combinations = xr.DataArray(
            list(odd_products),
            dims=["ind", "detector"],
            coords=dict(detector=probs.detector),
        )

        comb_probs = xr.where(combinations, probs.sel(state=1), probs.sel(state=0))

        stab_defect_probs = comb_probs.prod(dim="detector").sum(dim="ind")
        final_defect_probs[..., ind] = stab_defect_probs

    return final_defect_probs


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
    anc_meas = dataset.anc_meas
    data_meas = dataset.data_meas

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    return anc_meas, data_meas, log_errors


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
    syndromes = get_syndromes(anc_flips)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    final_syndromes = (data_flips @ proj_mat) % 2

    log_errors = data_flips.sum(dim="data_qubit") % 2

    return syndromes, final_syndromes, log_errors


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
    defects = get_defects(syndromes)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    proj_syndrome = (data_flips @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    log_errors = data_flips.sum(dim="data_qubit") % 2

    return defects, final_defects, log_errors


def to_model_input(
    recurrent_inputs: xr.DataArray,
    eval_inputs: xr.DataArray,
    log_errors: xr.DataArray,
    expansion_matrix: Optional[xr.DataArray] = None,
    data_type: Optional = bool,
):
    if expansion_matrix is not None:
        expanded_inputs = recurrent_inputs @ expansion_matrix
        recurrent_input = expanded_inputs.values.astype(data_type)
    else:
        recurrent_input = recurrent_inputs.values.astype(data_type)

    eval_input = eval_inputs.values.astype(data_type)

    inputs = dict(
        recurrent_input=recurrent_input,
        eval_input=eval_input,
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
    # Get Gaussian params
    means = np.array([-1, 1])
    dev = dev_from_error(means, float(dataset.error_prob))

    rng = np.random.default_rng(seed=int(dataset.seed))  # avoids TypeError

    samples = rng.normal(means, dev, size=(*dataset.anc_meas.shape, 2))
    anc_outcomes = xr.where(dataset.anc_meas, samples[..., 1], samples[..., 0])

    anc_probs = get_state_probs(anc_outcomes, means, dev)
    defect_probs = get_defect_probs(anc_probs)

    samples = rng.normal(means, dev, size=(*dataset.data_meas.shape, 2))
    data_outcomes = xr.where(dataset.data_meas, samples[..., 1], samples[..., 0])

    data_probs = get_state_probs(data_outcomes, means, dev)
    final_defect_probs = get_final_defect_probs(anc_probs, data_probs, proj_mat)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    return defect_probs, final_defect_probs, log_errors
