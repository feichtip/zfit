from typing import Iterable, Mapping

from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitData, ZfitPDF
from ..core.loss import BaseLoss, ExtendedUnbinnedNLL
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container
from .binnedloss import ExtendedBinnedNLL


class HybridLoss(BaseLoss):
    """A hybrid loss that combines binned and unbinned negative log-likelihood losses."""

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
            options: Additional options for the loss calculation
        """
        # Convert inputs to lists if they aren't already
        unbinned_model = convert_to_container(unbinned_model)
        unbinned_data = convert_to_container(unbinned_data)
        binned_model = convert_to_container(binned_model)
        binned_data = convert_to_container(binned_data)

        # Create the individual loss components
        self._unbinned_loss = ExtendedUnbinnedNLL(
            model=unbinned_model,
            data=unbinned_data,
            fit_range=fit_range,
            constraints=None,  # Constraints will be handled by the hybrid loss
            options=options
        )

        self._binned_loss = ExtendedBinnedNLL(
            model=binned_model,
            data=binned_data,
            constraints=None,  # Constraints will be handled by the hybrid loss
            options=options
        )

        # Initialize base class with combined models and data
        super().__init__(
            model=list(unbinned_model) + list(binned_model),
            data=list(unbinned_data) + list(binned_data),
            fit_range=fit_range,
            constraints=constraints,
            options=options
        )

        self._errordef = 0.5  # Same as other NLL losses

    @property
    def is_extended(self):
        return True

    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        # Split models and data back into binned and unbinned parts
        n_unbinned = len(self._unbinned_loss.model)
        unbinned_model = model[:n_unbinned]
        unbinned_data = data[:n_unbinned]
        binned_model = model[n_unbinned:]
        binned_data = data[n_unbinned:]

        # Calculate individual losses with the same log_offset
        unbinned_nll = self._unbinned_loss._loss_func(
            unbinned_model,
            unbinned_data,
            fit_range,
            None,  # Constraints handled at hybrid level
            log_offset
        )

        binned_nll = self._binned_loss._loss_func(
            binned_model,
            binned_data,
            None,  # Fit range not used in binned loss
            None,  # Constraints handled at hybrid level
            log_offset
        )

        # Combine losses
        total_nll = unbinned_nll + binned_nll

        # Add constraints if present
        if constraints:
            constraint_terms = sum(constraint.value() for constraint in constraints)
            total_nll += constraint_terms

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
