from typing import Iterable, Mapping

from .. import z
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitData, ZfitPDF
from ..core.loss import BaseLoss, ExtendedUnbinnedNLL
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container
from .binnedloss import ExtendedBinnedNLL


class HybridLoss(BaseLoss):
    """A hybrid loss that combines binned and unbinned negative log-likelihood losses
    with separate handling of offsets for numerical stability."""

    def __init__(
        self,
        unbinned_model: ZfitPDF | Iterable[ZfitPDF],
        unbinned_data: ZfitData | Iterable[ZfitData],
        binned_model: ZfitBinnedPDF | Iterable[ZfitBinnedPDF],
        binned_data: ZfitBinnedData | Iterable[ZfitBinnedData],
        fit_range=None,
        constraints=None,
        options: Mapping | None = None,
    ):
        """Initialize a hybrid loss combining binned and unbinned NLL.

        Args:
            unbinned_model: PDFs for the unbinned part of the fit
            unbinned_data: Unbinned datasets
            binned_model: PDFs for the binned part of the fit
            binned_data: Binned datasets
            fit_range: Fit range for the unbinned part
            constraints: Additional constraints on the parameters
            options: Additional options for the loss calculation. In addition to standard options:
                    - 'subtr_const': bool or {'unbinned': bool, 'binned': bool}
                      If bool: applies to both parts
                      If dict: separate control for each part
        """
        options = {} if options is None else dict(options)

        # Handle separate offset options
        subtr_const = options.pop('subtr_const', True)
        if isinstance(subtr_const, bool):
            self._subtr_const = {'unbinned': subtr_const, 'binned': subtr_const}
        elif isinstance(subtr_const, dict):
            self._subtr_const = subtr_const
        else:
            raise ValueError("subtr_const must be bool or dict")

        # Convert inputs to lists
        unbinned_model = convert_to_container(unbinned_model)
        unbinned_data = convert_to_container(unbinned_data)
        binned_model = convert_to_container(binned_model)
        binned_data = convert_to_container(binned_data)

        # Create individual loss components with their own offset settings
        unbinned_options = dict(options)
        unbinned_options['subtr_const'] = self._subtr_const['unbinned']
        self._unbinned_loss = ExtendedUnbinnedNLL(
            model=unbinned_model,
            data=unbinned_data,
            fit_range=fit_range,
            constraints=None,  # Constraints will be handled by the hybrid loss
            options=unbinned_options
        )

        binned_options = dict(options)
        binned_options['subtr_const'] = self._subtr_const['binned']
        self._binned_loss = ExtendedBinnedNLL(
            model=binned_model,
            data=binned_data,
            constraints=None,  # Constraints will be handled by the hybrid loss
            options=binned_options
        )

        # Initialize base class with combined models and data
        super().__init__(
            model=list(unbinned_model) + list(binned_model),
            data=list(unbinned_data) + list(binned_data),
            fit_range=fit_range,
            constraints=constraints,
            options=options
        )

        self._errordef = 0.5

    @property
    def is_extended(self):
        return True

    def check_precompile(self, *, params=None, force=False):
        """Override to handle separate offset initialization for binned and unbinned parts."""
        params, needs_compile = super().check_precompile(params=params, force=force)

        # Let individual losses handle their own precompilation
        if needs_compile:
            self._unbinned_loss.check_precompile(params=params, force=force)
            self._binned_loss.check_precompile(params=params, force=force)

        return params, needs_compile

    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        # Split models and data
        n_unbinned = len(self._unbinned_loss.model)
        unbinned_model = model[:n_unbinned]
        unbinned_data = data[:n_unbinned]
        binned_model = model[n_unbinned:]
        binned_data = data[n_unbinned:]

        # Get individual loss values with their own offsets
        # log_offset parameter is ignored as each component handles its own offset
        unbinned_nll = self._unbinned_loss._loss_func(
            unbinned_model,
            unbinned_data,
            fit_range,
            None,  # Constraints handled at hybrid level
            None  # Each loss handles its own offset
        )

        binned_nll = self._binned_loss._loss_func(
            binned_model,
            binned_data,
            None,  # Fit range not used in binned loss
            None,  # Constraints handled at hybrid level
            None  # Each loss handles its own offset
        )

        # Combine losses
        total_nll = unbinned_nll + binned_nll

        # Add constraints if present
        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            total_nll += constraints

        return total_nll

    def value(self, *, params: ztyping.ParamTypeInput = None, full: bool | None = None):
        """Calculate the loss value.

        Args:
            params: Parameters to evaluate the loss at
            full: If True, return the full loss without any offset subtraction
        """
        if full:
            # Temporarily disable offset subtraction
            old_subtr = self._subtr_const
            self._subtr_const = {'unbinned': False, 'binned': False}
            try:
                return super().value(params=params, full=full)
            finally:
                self._subtr_const = old_subtr
        return super().value(params=params, full=full)

    def create_new(
        self,
        unbinned_model: ztyping.PDFInputType = NONE,
        unbinned_data: ztyping.DataInputType = NONE,
        binned_model: ztyping.BinnedPDFInputType = NONE,
        binned_data: ztyping.BinnedDataInputType = NONE,
        fit_range=NONE,
        constraints=NONE,
        options=NONE,
    ):
        """Create a new HybridLoss with updated arguments."""
        if unbinned_model is NONE:
            unbinned_model = self._unbinned_loss.model
        if unbinned_data is NONE:
            unbinned_data = self._unbinned_loss.data
        if binned_model is NONE:
            binned_model = self._binned_loss.model
        if binned_data is NONE:
            binned_data = self._binned_loss.data
        if fit_range is NONE:
            fit_range = self.fit_range
        if constraints is NONE:
            constraints = self.constraints
        if options is NONE:
            options = dict(self._options)
            options['subtr_const'] = self._subtr_const

        return type(self)(
            unbinned_model=unbinned_model,
            unbinned_data=unbinned_data,
            binned_model=binned_model,
            binned_data=binned_data,
            fit_range=fit_range,
            constraints=constraints,
            options=options,
        )
