from typing import Generator, List

import xarray as xr


def dataset_generator(
    datasets_dir: str,
    experiment_name: str,
    basis: str,
    shots: int,
    states: List[str],
    rounds: List[int],
) -> Generator:
    for num_rounds in rounds:
        for state in states:
            experiment = experiment_name.format(
                basis=basis,
                state=state,
                shots=shots,
                num_rounds=num_rounds,
            )
            try:
                dataset = xr.open_dataset(datasets_dir / experiment / "measurements.nc")
            except FileNotFoundError as error:
                raise ValueError("Invalid experiment data directory") from error

            yield dataset
