#  Copyright (c) 2019 zfit
"""Recurrent polynomials."""

from typing import Callable, List

import tensorflow as tf
import numpy as np
from zfit import ztf
from zfit.util import ztyping

from ..core.limits import Space
from ..core.basepdf import BasePDF


class RecursivePolynomial(BasePDF):
    """1D polynomial generated via three-term recurrence.


    """

    def __init__(self, obs, coeffs: list,
                 apply_scaling: bool = True,
                 name: str = "Polynomial", **kwargs):  # noqa
        """

        Args:
            coeffs (list): Coefficients for each polynomial. Used to calculate the degree.
            apply_scaling (bool): Rescale the data so that the actual limits represent (-1, 1).
            Recurrence(func): Recurrence relation as function of the two previous
                polynomials and the degree n:

                .. math::
                   x_{n+1} = recurrence(x_{n}, x_{n-1}, n)

        """
        # 0th coefficient set to 1 by default
        coeffs.insert(0, ztf.constant(1.))
        params = {f"c_{i}": coeff for i, coeff in enumerate(coeffs)}
        self._degree = len(coeffs) - 1  # 1 coeff -> 0th degree
        self._do_scale = apply_scaling
        if apply_scaling and not (isinstance(obs, Space) and obs.n_limits == 1):
            raise ValueError("obs need to be a Space with exactly one limit if rescaling is requested.")
        super().__init__(obs=obs, name=name, params=params, **kwargs)

    def _polynomials_rescale(self, x):
        if self._do_scale:
            lim_low, lim_high = self.space.limit1d
            x = (2 * x - lim_low - lim_high) / (lim_high - lim_low)
        return x

    @property
    def degree(self):
        """int: degree of the polynomial."""
        return self._degree

    def _unnormalized_pdf(self, x):
        x = x.unstack_x()
        x = self._polynomials_rescale(x)
        return self._poly_func(x=x, coeffs=self.params)


def create_poly(x, polys, coeffs, recurrence):
    degree = len(coeffs) - 1
    polys = do_recurrence(x, polys=polys, degree=degree, recurrence=recurrence)
    sum_polys = tf.reduce_sum([coeffs[f"c_{i}"] * poly for i, poly in enumerate(polys)], axis=0)
    return sum_polys


def do_recurrence(x, polys, degree, recurrence):
    polys = [polys[0](x), polys[1](x)]
    for i_deg in range(1, degree):  # recurrence returns for the n+1th degree
        polys.append(recurrence(polys[-1], polys[-2], i_deg, x))
    return polys


legendre_polys = [lambda x: tf.ones_like(x), lambda x: x]


def legendre_recurrence(p1, p2, n, x):
    """Recurrence relation for Legendre polynomials.

    .. math::
         (n+1) P_{n+1}(x) = (2n + 1) x P_{n}(x) - n P_{n-1}(x)

    """
    return ((2 * n + 1) * tf.multiply(x, p1) - n * p2) / (n + 1)


def legendre_shape(x, coeffs):
    return create_poly(x=x, polys=legendre_polys, coeffs=coeffs, recurrence=legendre_recurrence)


def legendre_integral(limits: ztyping.SpaceType, norm_range: ztyping.SpaceType,
                      params: List["zfit.Parameter"], model: RecursivePolynomial):
    """Recursive integral of Legendre polynomials"""
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)
    if np.allclose((lower_rescaled, upper_rescaled), (-1, 1)):
        return ztf.constant(2.)  #

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    integral_0 = model.params[f"c_0"] * (upper - lower)  # if polynomial 0 is 1
    if model.degree == 0:
        integral = integral_0
    else:

        def indefinite_integral(limits):
            max_degree = model.degree + 1  # needed +1 for integral, max poly in term for n is n+1
            polys = do_recurrence(x=limits, polys=legendre_polys, degree=max_degree,
                                  recurrence=legendre_recurrence)
            one_limit_integrals = []
            for degree in range(1, max_degree):
                coeff = model.params[f"c_{degree}"]
                one_limit_integrals.append(coeff * (polys[degree + 1] - polys[degree - 1]) /
                                           (2. * (ztf.convert_to_tensor(degree)) + 1))
            return ztf.reduce_sum(one_limit_integrals, axis=0)

        integral = indefinite_integral(upper) - indefinite_integral(lower) + integral_0
        integral = tf.reshape(integral, shape=())

    return integral


class Legendre(RecursivePolynomial):
    """Legendre polynomials."""

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, name: str = "Legendre", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         coeffs=coeffs, apply_scaling=apply_scaling, **kwargs)

    @staticmethod
    def _poly_func(x, coeffs):
        return legendre_shape(x=x, coeffs=coeffs)


legendre_limits = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Legendre.register_analytic_integral(func=legendre_integral, limits=legendre_limits)

chebyshev_polys = [lambda x: tf.ones_like(x), lambda x: x]


def chebyshev_recurrence(p1, p2, _, x):
    """Recurrence relation for Chebyshev polynomials.

    T_{n+1}(x) = 2 x T_{n}(x) - T_{n-1}(x)

    """
    return 2 * tf.multiply(x, p1) - p2


def chebyshev_shape(x, coeffs):
    return create_poly(x=x, polys=chebyshev_polys, coeffs=coeffs, recurrence=chebyshev_recurrence)


class Chebyshev(RecursivePolynomial):
    """Chebyshev polynomials."""

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, name: str = "Chebyshev", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         coeffs=coeffs,
                         apply_scaling=apply_scaling, **kwargs)

    @staticmethod
    def _poly_func(x, coeffs):
        return chebyshev_shape(x=x, coeffs=coeffs)


def func_integral_chebyshev1(limits, norm_range, params, model):
    lower, upper = limits.limit1d
    lower_rescaled = model._polynomials_rescale(lower)
    upper_rescaled = model._polynomials_rescale(upper)
    # if np.allclose((lower_rescaled, upper_rescaled), (-1, 1)):
    #     return ztf.constant(2.)  #

    lower = ztf.convert_to_tensor(lower_rescaled)
    upper = ztf.convert_to_tensor(upper_rescaled)

    integral = model.params[f"c_0"] * (upper - lower)  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 1:
        integral += model.params[f"c_1"] * 0.5 * (upper ** 2 - lower ** 2)  # if polynomial 0 is defined as T_0 = 1
    if model.degree >= 2:

        def indefinite_integral(limits):
            max_degree = model.degree + 1
            polys = do_recurrence(x=limits, polys=chebyshev_polys, degree=max_degree,
                                  recurrence=chebyshev_recurrence)
            one_limit_integrals = []
            for degree in range(2, max_degree):
                coeff = model.params[f"c_{degree}"]
                n_float = ztf.convert_to_tensor(degree)
                integral = (n_float * polys[degree + 1] / (ztf.square(n_float) - 1)
                            - limits * polys[degree] / (n_float - 1))
                one_limit_integrals.append(coeff * integral)
            return ztf.reduce_sum(one_limit_integrals, axis=0)

        integral += indefinite_integral(upper) - indefinite_integral(lower)
        integral = tf.reshape(integral, shape=())

    return integral


chebyshev1_limits_integral = Space.from_axes(axes=0, limits=(Space.ANY_LOWER, Space.ANY_UPPER))
Chebyshev.register_analytic_integral(func=func_integral_chebyshev1, limits=chebyshev1_limits_integral)


class Chebyshev2(RecursivePolynomial):
    """Chebyshev polynomials of the second kind."""

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, name: str = "Chebyshev2", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: x * 2, coeffs=coeffs,
                         recurrence=chebyshev_recurrence, apply_scaling=apply_scaling, **kwargs)


def laguerre_recurrence(p1, p2, n, x):
    """Recurrence relation for Laguerre polynomials.

    (n+1) L_{n+1}(x) = (2n + 1 - x) L_{n}(x) - n L_{n-1}(x)

    """
    return (tf.multiply(2 * n + 1 - x, p1) - n * p2) / (n + 1)


class Laguerre(RecursivePolynomial):
    """Laguerre polynomials."""

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, name: str = "Laguerre", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: 1 - x, coeffs=coeffs,
                         recurrence=laguerre_recurrence, apply_scaling=apply_scaling, **kwargs)


def hermite_recurrence(p1, p2, n, x):
    """Recurrence relation for Hermite polynomials (physics).

    H_{n+1}(x) = 2x H_{n}(x) - 2n H_{n-1}(x)

    """
    return 2 * (tf.multiply(x, p1) - n * p2)


class Hermite(RecursivePolynomial):
    """Hermite polynomials as defined for Physics."""

    def __init__(self, obs, coeffs: list, apply_scaling: bool = True, name: str = "Hermite", **kwargs):  # noqa
        super().__init__(obs=obs, name=name,
                         f0=lambda x: tf.ones_like(x), f1=lambda x: 2 * x, coeffs=coeffs,
                         recurrence=hermite_recurrence, apply_scaling=apply_scaling, **kwargs)

# EOF
