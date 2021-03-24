#  Copyright (c) 2021 zfit
from typing import Iterable, Optional

import tensorflow as tf

from ..core.interfaces import ZfitIndependentParameter, ZfitLoss
from ..util.exception import OperationNotAllowedError
from .baseminimizer import BaseStepMinimizer, minimize_supports


class WrapOptimizer(BaseStepMinimizer):
    def __init__(self,
                 optimizer,
                 tol=None,
                 criterion=None,
                 strategy=None,
                 verbosity=None,
                 name=None,
                 **kwargs):
        """Wrap TensorFlow optimizers to have the zfit interface and behavior.

        .. note:: Different behavior of minimize

          While the `minimize` method in TensorFlow optimizers executes a single step of the minimization,
          the `minimize` method of a :class:`ZfitMinimizer` fully minimizes a function until convergence
          is reached. In order to execute a single step, use the :meth:`~WrapOptimizer.step` method (however,
          this is in general not necessary to do and rather inefficient).

        Args:
            optimizer: Instance of a :class:`tf.keras.optimizers.Optimizer`.
            All other arguments: these are passed through and have the same functionality as described
            in :py:class:`~BaseStepMinimizer`
        """
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            raise TypeError("optimizer {} has to be from class Optimizer".format(str(optimizer)))
        super().__init__(tol=tol, criterion=criterion, strategy=strategy, verbosity=verbosity, name=name,
                         minimizer_options=None, **kwargs)
        self._optimizer_tf = optimizer

    @minimize_supports(init=True)
    def _minimize(self, loss, params, init):
        try:
            return super()._minimize(loss, params, init)
        except ValueError as error:
            if 'No gradients provided for any variable' in error.args[0]:
                raise OperationNotAllowedError("Cannot use TF optimizer with"
                                               " a numerical gradient (non-TF function)") from None
            else:
                raise

    def _step(self,
              loss: ZfitLoss,
              params: Iterable[ZfitIndependentParameter],
              init: Optional["zfit.result.FitResult"]
              ) -> tf.Tensor:
        self._optimizer_tf.minimize(loss=loss.value, var_list=params)
        return loss.value()
