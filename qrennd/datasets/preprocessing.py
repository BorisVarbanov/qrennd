from typing import Optional

from xarray import DataArray, Dataset


def get_syndromes(anc_meas: DataArray, meas_reset: bool) -> DataArray:
    if meas_reset:
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
    anc_meas = dataset.anc_meas.transpose(..., "qec_round", "anc_qubit")
    data_meas = dataset.data_meas.transpose(..., "data_qubit")
    # due to stacking of "data_init" and "shot" in a common coordinate "run"
    # the ideal measurements have also "run" dimension
    ideal_anc_meas = dataset.ideal_anc_meas.transpose(..., "qec_round", "anc_qubit")
    ideal_data_meas = dataset.ideal_data_meas.transpose(..., "data_qubit")

    data_meas = data_meas ^ ideal_data_meas
    anc_meas = anc_meas ^ ideal_anc_meas

    inputs = {}

    if lstm_input == "measurements":
        # without ideal correction
        inputs["lstm_input"] = anc_meas.values ^ ideal_anc_meas
    elif lstm_input == "syndromes":
        syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)
        inputs["lstm_input"] = syndromes.values
    elif lstm_input == "defects":
        syndromes = get_syndromes(anc_meas, meas_reset=dataset.meas_reset.values)
        defects = get_defects(syndromes)
        inputs["lstm_input"] = defects.values
    else:
        raise ValueError(
            f"Unknown input data type {lstm_input}, the possible "
            "options are 'measurements', 'syndromes' and 'defects'."
        )

    if eval_input == "measurements":
        # without ideal correction
        inputs["eval_input"] = data_meas.values ^ ideal_data_meas
    elif eval_input == "syndromes":
        proj_syndrome = (data_meas @ proj_mat) % 2
        inputs["eval_input"] = proj_syndrome.values
    elif eval_input == "defects":
        proj_syndrome = (data_meas @ proj_mat) % 2
        final_defects = get_final_defects(syndromes, proj_syndrome)
        inputs["eval_input"] = final_defects.values
    else:
        raise ValueError(
            f"Unknown input data type {lstm_input}, the possible "
            "options are 'measurements', 'defects'."
        )

    log_errors = data_meas.sum(dim="data_qubit") % 2
    log_errors = log_errors.values

    return inputs, log_errors
