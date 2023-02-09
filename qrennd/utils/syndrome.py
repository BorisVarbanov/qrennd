from typing import Optional

from xarray import DataArray


def get_syndromes(anc_meas: DataArray, meas_reset: bool) -> DataArray:
    if meas_reset:
        syndromes = anc_meas
        syndromes.name = "syndromes"
    else:
        syndromes = anc_meas ^ anc_meas.shift(qec_round=1, fill_value=0)
        syndromes.name = "syndromes"
    return syndromes


def get_defects(syndromes: DataArray, frame: Optional[DataArray] = None) -> DataArray:
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    return defects


def get_final_defects(
    syndromes: DataArray,
    proj_syndrome: DataArray,
) -> DataArray:
    last_syndrome = syndromes.isel(qec_round=-1)
    proj_anc = proj_syndrome.anc_qubit

    final_defects = last_syndrome.sel(anc_qubit=proj_anc) ^ proj_syndrome
    return final_defects
