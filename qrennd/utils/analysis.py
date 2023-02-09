from typing import Union
import lmfit

import numpy as np
from numpy.typing import NDArray
from uncertainties import ufloat


def error_prob(predictions: NDArray, values: NDArray) -> float:
    return np.mean(predictions ^ values)


def logical_fidelity(predictions: NDArray, values: NDArray) -> float:
    return 1 - error_prob(predictions, values)


def error_prob_decay(
    qec_round: Union[int, NDArray], error_rate: float
) -> Union[int, NDArray]:
    return 0.5 * (1 - (1 - 2 * error_rate) ** qec_round)


def logical_fidelity_decay(
    qec_round: Union[int, NDArray], error_rate: float
) -> Union[int, NDArray]:
    return 1 - error_prob_decay(qec_round, error_rate)


class LogicalFidelityDecay(lmfit.model.Model):
    """
    lmfit model with a guess for a logical fidelity decay.
    """

    def __init__(self):
        # pass in the model's equation
        def funct(x, error_rate):
            return 0.5 + 0.5 * (1 - 2 * error_rate) ** x

        super().__init__(funct)

        # configure constraints that are independent from the data to be fitted
        self.set_param_hint("error_rate", min=0, max=1, vary=True)

    def guess(self, data: NDArray, x: NDArray, **kws) -> lmfit.parameter.Parameters:
        # to ensure they are np.ndarrays
        x, data = np.array(x), np.array(data)
        # guess parameters based on the data
        deriv_data = (data[1:] - data[:-1]) / (x[1:] - x[:-1])
        data_averaged = 0.5 * (data[1:] + data[:-1])
        error_rate_guess = 0.5 * (1 - np.exp(np.average(deriv_data / data_averaged)))

        self.set_param_hint("error_rate", value=error_rate_guess)
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def fit(
        self,
        data: NDArray,
        params: lmfit.parameter.Parameters,
        x: NDArray = None,
        min_qec: int = 0,
        **kws
    ):
        """
        Parameters
        ----------
        min_qec
            Minimum QEC round to perform the fit to
        """
        data = data[np.where(x >= min_qec)]
        x = x[np.where(x >= min_qec)]
        return super().fit(data, params, x=x, **kws)


def lmfit_par_to_ufloat(param: lmfit.parameter.Parameter):
    """
    Safe conversion of an :class:`lmfit.parameter.Parameter` to
    :code:`uncertainties.ufloat(value, std_dev)`.
    """

    value = param.value
    stderr = np.nan if param.stderr is None else param.stderr

    return ufloat(value, stderr)
