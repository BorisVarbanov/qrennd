from itertools import product
from typing import Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.special import erfcinv


def odd_parity(bits: Sequence[int]) -> int:
    return sum(bits) % 2


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


def to_defect_probs_leakage_IQ(
    dataset: xr.Dataset,
    proj_mat: xr.DataArray,
    digitization: Optional[dict] = {"data": False, "anc": False},
    leakage: Optional[dict] = {"data": False, "anc": False},
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
    proj_mat
        Assumes to have dimensions [data_qubits, stab],
        where stab correspond to the final stabilizers.
    digitization
        Flag for digitizing the defect probability
    """
    anc_probs, data_probs = get_state_probs_IQ(dataset)

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

    data_meas = data_probs.sel(state=1) > data_probs.sel(state=0)
    data_meas = data_meas.transpose("shot", "data_qubit")
    data_flips = data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    if digitization["anc"]:
        defect_probs = defect_probs > 0.5
    if digitization["data"]:
        final_defect_probs = final_defect_probs > 0.5

    # add leakage outcomes
    rec_inputs, eval_inputs = defect_probs, final_defect_probs
    if leakage["anc"]:
        anc_leakage_flag = dataset.anc_leakage_flag
        rec_inputs = [defect_probs, anc_leakage_flag]
    if leakage["data"]:
        data_leakage_flag = dataset.data_leakage_flag
        eval_inputs = [final_defect_probs, data_leakage_flag]

    return (
        rec_inputs,
        eval_inputs,
        log_errors,
    )


def get_state_probs_IQ(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    Calculates the probabilities of the qubit
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

    # projection to 1d
    def project(x, theta):
        # the projection need to be done clockwise
        rot_mat = xr.DataArray(
            [np.cos(theta), np.sin(theta)], coords=dict(iq=["I", "Q"])
        )
        z = x @ rot_mat
        return z

    # pdf that 'iq_readout' repository uses for fitting
    def simple_1d_gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        z = (
            1
            / np.sqrt(2 * np.pi * sigma**2)
            * np.exp(-0.5 * (x - mu) ** 2 / sigma**2)
        )
        return z

    def simple_1d_gaussian_double_mixture(
        x: np.ndarray, mu_0: float, mu_1: float, sigma: float, angle: float
    ) -> np.ndarray:
        a1, a2 = np.sin(angle) ** 2, np.cos(angle) ** 2
        z = a1 * simple_1d_gaussian(x, mu=mu_0, sigma=sigma) + a2 * simple_1d_gaussian(
            x, mu=mu_1, sigma=sigma
        )
        return z

    param_names = ["mu_0", "mu_1", "sigma", "angle"]

    # data qubits
    probs_0_list = []
    for qubit in dataset.data_qubit:
        outcomes = dataset.data_meas.sel(data_qubit=qubit)
        outcomes = project(outcomes, dataset.rot_angles.sel(qubit=qubit))
        params = {
            param: dataset.pdf_0_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = simple_1d_gaussian_double_mixture(outcomes, **params)
        probs_0_list.append(probs)
    probs_0_list = xr.concat(probs_0_list, dim="data_qubit")

    probs_1_list = []
    for qubit in dataset.data_qubit:
        outcomes = dataset.data_meas.sel(data_qubit=qubit)
        outcomes = project(outcomes, dataset.rot_angles.sel(qubit=qubit))
        params = {
            param: dataset.pdf_1_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = simple_1d_gaussian_double_mixture(outcomes, **params)
        probs_1_list.append(probs)
    probs_1_list = xr.concat(probs_1_list, dim="data_qubit")
    data_probs = xr.concat([probs_0_list, probs_1_list], dim="state")

    # ancilla qubits
    probs_0_list = []
    for qubit in dataset.anc_qubit:
        outcomes = dataset.anc_meas.sel(anc_qubit=qubit)
        outcomes = project(outcomes, dataset.rot_angles.sel(qubit=qubit))
        params = {
            param: dataset.pdf_0_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = simple_1d_gaussian_double_mixture(outcomes, **params)
        probs_0_list.append(probs)
    probs_0_list = xr.concat(probs_0_list, dim="anc_qubit")

    probs_1_list = []
    for qubit in dataset.anc_qubit:
        outcomes = dataset.anc_meas.sel(anc_qubit=qubit)
        outcomes = project(outcomes, dataset.rot_angles.sel(qubit=qubit))
        params = {
            param: dataset.pdf_1_params.sel(qubit=qubit, param=param)
            for param in param_names
        }
        probs = simple_1d_gaussian_double_mixture(outcomes, **params)
        probs_1_list.append(probs)
    probs_1_list = xr.concat(probs_1_list, dim="anc_qubit")
    anc_probs = xr.concat([probs_0_list, probs_1_list], dim="state")

    anc_probs = anc_probs / anc_probs.sum(dim="state")
    data_probs = data_probs / data_probs.sum(dim="state")

    anc_probs = anc_probs.transpose("state", "shot", "qec_round", "anc_qubit")
    data_probs = data_probs.transpose("state", "shot", "data_qubit")

    return anc_probs, data_probs
