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


def preprocess_data(
    dataset: Dataset,
    lstm_input: str,
    eval_input: str,
    proj_mat: Optional[DataArray] = None,
):
    anc_meas = dataset.anc_meas
    data_meas = dataset.data_meas

    ideal_anc_meas = dataset.ideal_anc_meas
    ideal_data_meas = dataset.ideal_data_meas

    if lstm_input == "measurements":
        lstm_inputs = anc_meas
    elif lstm_input == "syndromes":
        anc_flips = anc_meas ^ ideal_anc_meas
        lstm_inputs = get_syndromes(anc_flips)
    elif lstm_input == "defects":
        anc_flips = anc_meas ^ ideal_anc_meas
        syndromes = get_syndromes(anc_flips)
        lstm_inputs = get_defects(syndromes)
    else:
        raise ValueError(
            f"Unknown input data type {lstm_input}, the possible "
            "options are 'measurements', 'syndromes' and 'defects'."
        )
    lstm_inputs = lstm_inputs.stack(run=["init", "shot"])
    lstm_inputs = lstm_inputs.transpose("run", "qec_round", "anc_qubit")

    if eval_input == "measurements":
        eval_inputs = dataset.data_meas
    elif eval_input == "syndromes":
        data_flips = data_meas ^ ideal_data_meas
        eval_inputs = (data_meas @ proj_mat) % 2
    elif eval_input == "defects":
        data_flips = data_meas ^ ideal_data_meas
        proj_syndrome = (data_flips @ proj_mat) % 2
        eval_inputs = get_final_defects(syndromes, proj_syndrome)
    else:
        raise ValueError(
            f"Unknown input data type {lstm_input}, the possible "
            "options are 'measurements', 'defects'."
        )
    eval_inputs = eval_inputs.stack(run=["init", "shot"])
    eval_inputs = eval_inputs.transpose("run", ...)

    data_errors = data_meas ^ ideal_data_meas
    log_errors = data_errors.sum(dim="data_qubit") % 2
    log_errors = log_errors.stack(run=["init", "shot"])

    inputs = dict(
        lstm_input=lstm_inputs.values.astype(bool),
        eval_input=eval_inputs.values.astype(bool),
    )
    outputs = log_errors.values.astype(bool)

    return inputs, outputs
