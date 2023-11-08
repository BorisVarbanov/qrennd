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


def get_state_probs_experimental(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    get_state_probs Calculates the probabilities of the qubit
    being in a given state (0 or 1) given the soft measurement
    outcomes for the ancilla and the data qubits.

    Parameters
    ----------
    outcomes : xr.Dataset
        The soft measurement outcomes.

    Returns
    -------
    anc_probs: xr.Dataset
        The probabilities of the ancilla qubits being in each state given the outcomes.
    data_probs: xr.Dataset
        The probabilities of the data qubits being in each state given the outcomes.
    """

    # pdf that DiCarlo uses for fitting
    def _gauss_pdf(x, x0, sigma):
        return np.exp(-(((x - x0) / sigma) ** 2) / 2) / (np.sqrt(2 * np.pi) * sigma)

    def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
        _dist0 = (1 - r) * _gauss_pdf(x, x0, sigma0) + r * _gauss_pdf(x, x1, sigma1)
        return _dist0

    param_names = ["x0", "x1", "sigma0", "sigma1", "A", "r"]

    # data qubits
    probs_0_list = []
    for qubit in dataset.data_qubit:
        outcomes = dataset.data_meas.sel(data_qubit=qubit)
        params = {
            param: dataset.pdf_0_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = double_gauss(outcomes, **params)
        probs_0_list.append(probs)
    probs_0_list = xr.concat(probs_0_list, dim="data_qubit")

    probs_1_list = []
    for qubit in dataset.data_qubit:
        outcomes = dataset.data_meas.sel(data_qubit=qubit)
        params = {
            param: dataset.pdf_1_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = double_gauss(outcomes, **params)
        probs_1_list.append(probs)
    probs_1_list = xr.concat(probs_1_list, dim="data_qubit")
    data_probs = xr.concat([probs_0_list, probs_1_list], dim="state")

    # ancilla qubits
    probs_0_list = []
    for qubit in dataset.anc_qubit:
        outcomes = dataset.anc_meas.sel(anc_qubit=qubit)
        params = {
            param: dataset.pdf_0_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = double_gauss(outcomes, **params)
        probs_0_list.append(probs)
    probs_0_list = xr.concat(probs_0_list, dim="anc_qubit")

    probs_1_list = []
    for qubit in dataset.anc_qubit:
        outcomes = dataset.anc_meas.sel(anc_qubit=qubit)
        params = {
            param: dataset.pdf_1_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = double_gauss(outcomes, **params)
        probs_1_list.append(probs)
    probs_1_list = xr.concat(probs_1_list, dim="anc_qubit")
    anc_probs = xr.concat([probs_0_list, probs_1_list], dim="state")

    anc_probs = anc_probs / anc_probs.sum(dim="state")
    data_probs = data_probs / data_probs.sum(dim="state")

    anc_probs = anc_probs.transpose("state", "shot", "qec_round", "anc_qubit")
    data_probs = data_probs.transpose("state", "shot", "data_qubit")

    return anc_probs, data_probs


def get_defect_probs(
    anc_probs: xr.DataArray, ideal_defects: xr.DataArray
) -> xr.DataArray:
    """
    get_defect_probs Calculates the probability of observing a defect, given
    the probabilities of the ancilla qubits being in a given state (0 or 1).

    Parameters
    ----------
    anc_probs : xr.DataArray
        The probabilities of each ancilla qubits being in a given state (0 or 1) over
        each round of the experiment.
    ideal_defects : xr.DataArray
        Defect values when the circuit is executed without noise

    Returns
    -------
    xr.DataArray
        The probabilities of observing a defect at each round.
    """
    round_shift = 1 if anc_probs.meas_reset else 2

    shifted_probs = anc_probs.shift(qec_round=round_shift)
    prob_product = anc_probs.dot(shifted_probs, dims="state")
    defect_probs = 1 - prob_product.fillna(anc_probs.sel(state=0))
    defect_probs = xr.where(ideal_defects, 1 - defect_probs, defect_probs)
    # reshape into (shots, qec_round, anc_qubit) because ideal_defects does
    # not have "shot" dimension and messes the order of the coordinates
    defect_probs = defect_probs.transpose("shot", "qec_round", "anc_qubit")

    return defect_probs


def get_final_defect_probs(
    anc_probs: xr.DataArray,
    data_probs: xr.DataArray,
    ideal_final_defects: xr.DataArray,
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
    ideal_final_defects : xr.DataArray
        Final defect values when the circuit is executed without noise
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

    final_defect_probs = xr.where(
        ideal_final_defects, 1 - final_defect_probs, final_defect_probs
    )
    # reshape into (shots, qec_round, anc_qubit) because ideal_defects does
    # not have "shot" dimension and messes the order of the coordinates
    final_defect_probs = final_defect_probs.transpose("shot", "anc_qubit")

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
    # set leakage shots to |1>
    anc_meas = xr.where(dataset.anc_meas == 2, 1, dataset.anc_meas).astype(bool)
    data_meas = xr.where(dataset.data_meas == 2, 1, dataset.data_meas).astype(bool)

    anc_flips = anc_meas ^ dataset.ideal_anc_meas
    syndromes = get_syndromes(anc_flips)
    defects = get_defects(syndromes)

    data_flips = data_meas ^ dataset.ideal_data_meas
    proj_syndrome = (data_flips @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    log_errors = data_flips.sum(dim="data_qubit") % 2

    return defects, final_defects, log_errors


def to_model_input(
    rec_inputs: xr.DataArray,
    eval_inputs: xr.DataArray,
    log_errors: xr.DataArray,
    expansion_matrix: Optional[xr.DataArray] = None,
    data_type: type = bool,
):
    if expansion_matrix is not None:
        expanded_inputs = rec_inputs @ expansion_matrix
        rec_tensor = expanded_inputs.values.astype(data_type)
    else:
        if isinstance(rec_inputs, list):
            rec_tensor = [r.values.astype(data_type) for r in rec_inputs]
            rec_tensor = np.concatenate(rec_tensor, axis=2)
        else:
            rec_tensor = rec_inputs.values.astype(data_type)

    if isinstance(eval_inputs, list):
        eval_tensor = [r.values.astype(data_type) for r in eval_inputs]
        eval_tensor = np.concatenate(eval_tensor, axis=1)
    else:
        eval_tensor = eval_inputs.values.astype(data_type)

    error_tensor = log_errors.values.astype(data_type)

    return rec_tensor, eval_tensor, error_tensor


def to_defect_probs(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    assign_errors: dict,
    digitization: Optional[bool] = False,
):
    """
    Preprocess dataset to generate the probability of defect
    based on the soft outcomes and the logical errors.

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
    assign_errors
        Assignment error probability for the soft measurements.
        Should be of the form:
        {"anc": assign_error_ancilla,
         "data": assign_error_data}
    digitization
        Flag for digitizing the defect probability
    """
    # Get Gaussian params
    means = np.array([-1, 1])
    dev_anc = dev_from_error(means, assign_errors["anc"])
    dev_data = dev_from_error(means, assign_errors["data"])

    rng = np.random.default_rng(seed=int(dataset.seed))  # avoids TypeError

    samples = rng.normal(means, dev_anc, size=(*dataset.anc_meas.shape, 2))
    anc_outcomes = xr.where(dataset.anc_meas, samples[..., 1], samples[..., 0])

    anc_probs = get_state_probs(anc_outcomes, means, dev_anc)
    ideal_syndromes = get_syndromes(dataset.ideal_anc_meas)
    ideal_defects = get_defects(ideal_syndromes)
    defect_probs = get_defect_probs(anc_probs, ideal_defects)

    samples = rng.normal(means, dev_data, size=(*dataset.data_meas.shape, 2))
    data_outcomes = xr.where(dataset.data_meas, samples[..., 1], samples[..., 0])

    data_probs = get_state_probs(data_outcomes, means, dev_data)
    ideal_proj_syndrome = (dataset.ideal_data_meas @ proj_mat) % 2
    ideal_final_defects = get_final_defects(ideal_syndromes, ideal_proj_syndrome)
    final_defect_probs = get_final_defect_probs(
        anc_probs,
        data_probs,
        ideal_final_defects=ideal_final_defects,
        proj_mat=proj_mat,
    )

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    if digitization:
        defect_probs = defect_probs > 0.5
        final_defect_probs = final_defect_probs > 0.5

    return defect_probs, final_defect_probs, log_errors


def to_defect_probs_experimental(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    digitization: Optional[Tuple[bool, dict]] = False,
):
    """
    Preprocess dataset to generate the probability of defect
    based on the soft outcomes and the logical errors.

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
    digitization
        Flag for digitizing the defect probability
    """
    if isinstance(digitization, bool):
        digitization = {"anc": digitization, "data": digitization}

    anc_probs, data_probs = get_state_probs_experimental(dataset)

    ideal_syndromes = get_syndromes(dataset.ideal_anc_meas)
    ideal_defects = get_defects(ideal_syndromes)
    defect_probs = get_defect_probs(anc_probs, ideal_defects)

    ideal_proj_syndrome = (dataset.ideal_data_meas @ proj_mat) % 2
    ideal_final_defects = get_final_defects(ideal_syndromes, ideal_proj_syndrome)
    final_defect_probs = get_final_defect_probs(
        anc_probs,
        data_probs,
        ideal_final_defects=ideal_final_defects,
        proj_mat=proj_mat,
    )

    data_meas = digitize_final_measurements(dataset)
    data_flips = data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    if digitization["anc"]:
        defect_probs = defect_probs > 0.5
    if digitization["data"]:
        final_defect_probs = final_defect_probs > 0.5

    return defect_probs, final_defect_probs, log_errors


def digitize_final_measurements(dataset: xr.Dataset) -> xr.Dataset:
    """
    TODO
    """
    digitized_list = []
    for qubit in dataset.data_qubit:
        outcomes = dataset.data_meas.sel(data_qubit=qubit)
        threshold = dataset.thresholds.sel(qubit=qubit)
        params = dataset.pdf_0_params.sel(qubit=qubit).values
        mu0, mu1 = params[:2]
        if mu0 > mu1:
            digitized = outcomes < threshold
        else:
            digitized = outcomes > threshold
        digitized_list.append(digitized)
    digitized_list = xr.concat(digitized_list, dim="data_qubit")
    return digitized_list


def to_defect_probs_leakage_experimental(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    digitization: Optional[Tuple[bool, dict]] = False,
):
    """
    Preprocess dataset to generate the probability of defect
    based on the soft outcomes and the logical errors.

    Parameters
    ----------
    dataset
        Assumes to have the following variables and dimensions:
        - anc_meas: [shots, qec_cycle, anc_qubit]
        - ideal_anc_meas: [qec_cycle, anc_qubit]
        - data_meas: [shot, data_qubit]
        - anc_leakage_flag: [shots, qec_cycle, anc_qubit]
        - data_leakage_flag: [shot, data_qubit]
        - idea_data_meas: [data_qubit]
        - prob_error: float
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    digitization
        Flag for digitizing the defect probability
    """
    if isinstance(digitization, bool):
        digitization = {"anc": digitization, "data": digitization}

    anc_probs, data_probs = get_state_probs_experimental(dataset)

    ideal_syndromes = get_syndromes(dataset.ideal_anc_meas)
    ideal_defects = get_defects(ideal_syndromes)
    defect_probs = get_defect_probs(anc_probs, ideal_defects)

    ideal_proj_syndrome = (dataset.ideal_data_meas @ proj_mat) % 2
    ideal_final_defects = get_final_defects(ideal_syndromes, ideal_proj_syndrome)
    final_defect_probs = get_final_defect_probs(
        anc_probs,
        data_probs,
        ideal_final_defects=ideal_final_defects,
        proj_mat=proj_mat,
    )

    data_meas = digitize_final_measurements(dataset)
    data_flips = data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    if digitization["anc"]:
        defect_probs = defect_probs > 0.5
    if digitization["data"]:
        final_defect_probs = final_defect_probs > 0.5

    # add leakage outcomes
    anc_leakage_flag = dataset.anc_leakage_flag
    data_leakage_flag = dataset.data_leakage_flag

    return (
        [defect_probs, anc_leakage_flag],
        [final_defect_probs, data_leakage_flag],
        log_errors,
    )
