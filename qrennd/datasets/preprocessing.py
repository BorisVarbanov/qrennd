from typing import Optional

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
    anc_meas = dataset.anc_meas
    data_meas = dataset.data_meas

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    log_errors = data_flips.sum(dim="data_qubit") % 2

    return anc_meas, data_meas, log_errors


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
    syndromes = get_syndromes(anc_flips)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    final_syndromes = (data_flips @ proj_mat) % 2

    log_errors = data_flips.sum(dim="data_qubit") % 2

    return syndromes, final_syndromes, log_errors


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
    defects = get_defects(syndromes)

    data_flips = dataset.data_meas ^ dataset.ideal_data_meas
    proj_syndrome = (data_flips @ proj_mat) % 2
    final_defects = get_final_defects(syndromes, proj_syndrome)

    log_errors = data_flips.sum(dim="data_qubit") % 2

    return defects, final_defects, log_errors


def to_model_input(
    lstm_inputs: DataArray,
    eval_inputs: DataArray,
    log_errors: DataArray,
    expansion_matrix: Optional[DataArray] = None,
):
    if expansion_matrix is not None:
        expanded_inputs = lstm_inputs @ expansion_matrix
        lstm_input = expanded_inputs.values.astype(bool)
    else:
        lstm_input = lstm_input.values.astype(bool)

    eval_input = eval_inputs.values.astype(bool)

    inputs = dict(
        lstm_input=lstm_input,
        eval_input=eval_input,
    )
    outputs = log_errors.values.astype(bool)

    return inputs, outputs
