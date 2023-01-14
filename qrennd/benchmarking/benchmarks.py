from typing import List

import numpy as np
import lmfit


def log_fidelity(
    predictions: np.ndarray,
    true_value: np.ndarray,
) -> float:
    """Returns the logical fidelity.

    Parameters
    ----------
    predictions
        Booleans that the decoder predicted.
        It is a vector of the number of shots/samples.
    true_value
        Booleans of the true value that the decoder should predict
        It is a vector of the number of shots/samples.
    """
    return np.average(predictions == true_value)


def log_error_rate(qec_rounds: List[int], log_fidelity: List[float]):
    """Returns the logical error rate per cycle.

    Parameters
    ----------
    qec_rounds
        List of integers specifying the number of QEC rounds.
    log_fidelity
        List of the logical fidelity given a number of QEC rounds.
    """

    model = LogicalErrorRateDecay()
    pars = model.guess(log_fidelity, x=qec_rounds)
    out = model.fit(log_fidelity, pars, x=qec_rounds)
    eps = out.params["eps"].value
    return eps


class LogicalErrorRateDecay(lmfit.model.Model):
    """
    lmfit model with a guess for a logical error rate decay.
    """

    def __init__(self):
        # pass in the model's equation
        def funct(x, eps, t0):
            return 0.5 + 0.5 * (1 - 2 * eps) ** (x - t0)

        super().__init__(funct)

        # configure constraints that are independent from the data to be fitted
        self.set_param_hint("eps", min=0, max=1, vary=True)
        self.set_param_hint("t0", min=0, vary=True)

    def guess(self, data, x=None, **kws) -> lmfit.parameter.Parameters:
        # to ensure they are np.ndarrays
        data = np.array(data)
        x = np.array(x)
        # guess parameters based on the data
        deriv_data = (data[1:] - data[:-1]) / (x[1:] - x[:-1])
        data_averaged = 0.5 * (data[1:] + data[:-1])
        eps_guess = 0.5 * (1 - np.exp(np.average(deriv_data / data_averaged)))
        t0_guess = -np.log(2 * data[0] - 1) / np.log(1 - 2 * eps_guess)

        self.set_param_hint("eps", value=eps_guess)
        self.set_param_hint("t0", value=t0_guess)
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kws)
