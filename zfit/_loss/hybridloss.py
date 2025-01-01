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
            options: Additional options for the loss calculation.
                    'subtr_const': Controls offset subtraction for both parts
        """

        # Convert inputs to lists
        unbinned_model = convert_to_container(unbinned_model)
        unbinned_data = convert_to_container(unbinned_data)
        binned_model = convert_to_container(binned_model)
        binned_data = convert_to_container(binned_data)

        # Create individual loss components
        self._unbinned_loss = ExtendedUnbinnedNLL(
            model=unbinned_model,
            data=unbinned_data,
            fit_range=fit_range,
            constraints=None,  # Constraints handled by hybrid loss
            options=options
        )

        self._binned_loss = ExtendedBinnedNLL(
            model=binned_model,
            data=binned_data,
            constraints=None,  # Constraints handled by hybrid loss
            options=options
        )

        # Initialize base class
        super().__init__(
            model=list(unbinned_model) + list(binned_model),
            data=list(unbinned_data) + list(binned_data),
            fit_range=fit_range,
            constraints=constraints,
            options=options
        )

        self._errordef = 0.5
        self._offsets = {'unbinned': 0.0, 'binned': 0.0}

    @property
    def is_extended(self):
        return True

    def check_precompile(self, *, params=None, force=False):
        """Override to handle separate offset initialization for binned and unbinned parts."""
        params, needs_compile = super().check_precompile(params=params, force=force)

        if needs_compile and self._options.get("subtr_const", False):
            # set offset for HybridLoss to 0 (not used anywhere, so not needed)
            # self._options["subtr_const_value"] = 0.0

            # Calculate individual offsets
            self._unbinned_loss.check_precompile(params=params, force=force)
            self._binned_loss.check_precompile(params=params, force=force)

            # Store individual offsets
            self._offsets['unbinned'] = self._unbinned_loss._options.get("subtr_const_value", 0.0)
            self._offsets['binned'] = self._binned_loss._options.get("subtr_const_value", 0.0)

        return params, needs_compile

    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        # Split models and data
        n_unbinned = len(self._unbinned_loss.model)
        unbinned_model = model[:n_unbinned]
        unbinned_data = data[:n_unbinned]
        binned_model = model[n_unbinned:]
        binned_data = data[n_unbinned:]

        # Pass appropriate offset to each loss
        if log_offset is False:
            unbinned_offset = False
            binned_offset = False
        else:
            unbinned_offset = self._offsets['unbinned']
            binned_offset = self._offsets['binned']

        # Calculate individual losses with their respective offsets
        unbinned_nll = self._unbinned_loss._loss_func(
            unbinned_model,
            unbinned_data,
            fit_range,
            None,  # Constraints handled at hybrid level
            unbinned_offset
        )

        binned_nll = self._binned_loss._loss_func(
            binned_model,
            binned_data,
            None,  # Fit range not used in binned loss
            None,  # Constraints handled at hybrid level
            binned_offset
        )

        # Combine losses
        total_nll = unbinned_nll + binned_nll

        # Add constraints if present
        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            total_nll += constraints

        return total_nll

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
            options = self._options

        return type(self)(
            unbinned_model=unbinned_model,
            unbinned_data=unbinned_data,
            binned_model=binned_model,
            binned_data=binned_data,
            fit_range=fit_range,
            constraints=constraints,
            options=options,
        )
