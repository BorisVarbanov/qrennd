from typing import Union
import lmfit

import numpy as np
from numpy import ndarray
from uncertainties import ufloat


def error_prob(predictions: ndarray, values: ndarray) -> float:
    return np.mean(predictions ^ values)


def error_prob_decay(
    x: Union[int, ndarray],
    error_rate: float,
    t0: float,
) -> Union[int, ndarray]:
    return 0.5 - 0.5 * (1 - 2 * error_rate) ** (x - t0)


class LogicalErrorProb(lmfit.model.Model):
    """
    lmfit model with a guess for a logical fidelity decay.
    """

    def __init__(self, fixed_t0=True):
        super().__init__(error_prob_decay)
        self.fixed_t0 = fixed_t0

        # configure constraints that are independent from the data to be fitted
        self.set_param_hint("error_rate", min=0, max=1, vary=True)
        if self.fixed_t0:
            self.set_param_hint("t0", value=0, vary=False)
        else:
            self.set_param_hint("t0", vary=True)

    def guess(self, data: ndarray, x: ndarray, **kws) -> lmfit.parameter.Parameters:
        # to ensure they are np.ndarrays
        x, data = np.array(x), np.array(data)
        # guess parameters based on the data
        deriv_data = (data[1:] - data[:-1]) / (x[1:] - x[:-1])
        data_averaged = 0.5 * (data[1:] + data[:-1])
        error_rate_guess = 0.5 * (
            1 - np.exp(np.average(deriv_data / (data_averaged - 0.5)))
        )

        self.set_param_hint("error_rate", value=error_rate_guess)
        if not self.fixed_t0:
            self.set_param_hint("t0", value=0.01)
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kws)

    def fit(
        self,
        data: ndarray,
        params: lmfit.parameter.Parameters,
        x: ndarray,
        min_qec: int = 1,
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
