from typing import Iterable, Mapping

from .. import z
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitData, ZfitPDF
from ..core.loss import BaseLoss, ExtendedUnbinnedNLL, one_two_many
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container
from .._data.binneddatav1 import BinnedData
from .binnedloss import ExtendedBinnedNLL


class HybridLoss(BaseLoss):
    """A hybrid loss that combines binned and unbinned negative log-likelihood losses
    with separate handling of offsets for numerical stability."""

    def __init__(
        self,
        binned_model: ZfitBinnedPDF | Iterable[ZfitBinnedPDF],
        binned_data: ZfitBinnedData | Iterable[ZfitBinnedData],
        unbinned_model: ZfitPDF | Iterable[ZfitPDF],
        unbinned_data: ZfitData | Iterable[ZfitData],
        fit_range=None,
        constraints=None,
        options: Mapping | None = None,
    ):
        """Initialize a hybrid loss combining binned and unbinned NLL.

        Args:
            binned_model: PDFs for the binned part of the fit
            binned_data: Binned datasets
            unbinned_model: PDFs for the unbinned part of the fit
            unbinned_data: Unbinned datasets
            fit_range: Fit range for the unbinned part
            constraints: Additional constraints on the parameters
            options: Additional options for the loss calculation. In addition to standard options:
                    - 'subtr_const_unbinned': bool, default True
                    - 'subtr_const_binned': bool, default False
                    - 'subtr_const': bool, overwrites defaults of binned/unbined parts when set,
                                     no subtractions for both parts when set to False
        """
        options = {} if options is None else dict(options)

        # Set defaults of separate offset options
        default_subtr_const_binned = options.get('subtr_const', False)
        default_subtr_const_unbinned = options.get('subtr_const', True)

        # Convert inputs to lists
        binned_model = convert_to_container(binned_model)
        binned_data = convert_to_container(binned_data)
        unbinned_model = convert_to_container(unbinned_model)
        unbinned_data = convert_to_container(unbinned_data)

        # Create individual loss components with their own offset settings
        binned_options = dict(options)
        binned_options['subtr_const'] = options.get('subtr_const_binned', default_subtr_const_binned)
        self._binned_loss = ExtendedBinnedNLL(
            model=binned_model,
            data=binned_data,
            constraints=None,  # Constraints handled by hybrid loss
            options=binned_options
        )

        unbinned_options = dict(options)
        unbinned_options['subtr_const'] = options.get('subtr_const_unbinned', default_subtr_const_unbinned)
        self._unbinned_loss = ExtendedUnbinnedNLL(
            model=unbinned_model,
            data=unbinned_data,
            fit_range=fit_range,
            constraints=None,  # Constraints handled by hybrid loss
            options=unbinned_options
        )

        # Initialize base class
        super().__init__(
            model=list(binned_model) + list(unbinned_model),
            data=list(binned_data) + list(unbinned_data),
            fit_range=fit_range,
            constraints=constraints,
            options=options
        )

        self._errordef = 0.5
        self._offsets = {'binned': False, 'unbinned': False}

    def __repr__(self) -> str:
        class_name = repr(self.__class__)[:-2].split(".")[-1]
        data_nevents = [d._approx_nevents.numpy() for d in self.data]
        data_bins = [[len(bins) for bins in d.binning] if isinstance(d, BinnedData) else None for d in self.data]
        return (
            f"<{class_name} "
            f"model={[model.name or model.label for model in self.model]} "
            f"data={[data.name or data.label for data in self.data]} "
            f"nevents={data_nevents} "
            f"bins={data_bins} "
            f'constraints={one_two_many(self.constraints, many="True")} '
            f">"
        )

    @property
    def is_extended(self):
        return True

    def check_precompile(self, *, params=None, force=False):
        """Override to handle separate offset initialization for binned and unbinned parts."""
        params, needs_compile = super().check_precompile(params=params, force=force)

        if needs_compile and self._options["subtr_const"]:
            # set offset for HybridLoss to 0 (actual value not used anywhere, only should not be set to False)
            # self._options["subtr_const_value"] = 0.0

            # Calculate individual offsets
            self._binned_loss.check_precompile(params=params, force=force)
            self._unbinned_loss.check_precompile(params=params, force=force)

            # Store individual offsets
            self._offsets['binned'] = self._binned_loss._options.get("subtr_const_value", False)
            self._offsets['unbinned'] = self._unbinned_loss._options.get("subtr_const_value", False)

        return params, needs_compile

    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        # Split models and data
        n_binned = len(self._binned_loss.model)
        binned_model = model[:n_binned]
        binned_data = data[:n_binned]
        unbinned_model = model[n_binned:]
        unbinned_data = data[n_binned:]

        # Pass appropriate offset to each loss
        if log_offset is False:
            binned_offset = False
            unbinned_offset = False
        else:
            binned_offset = self._offsets['binned']
            unbinned_offset = self._offsets['unbinned']

        binned_nll = self._binned_loss._loss_func(
            binned_model,
            binned_data,
            None,  # Fit range not used in binned loss
            None,  # Constraints handled at hybrid level
            binned_offset
        )

        # Calculate individual losses with their respective offsets
        unbinned_nll = self._unbinned_loss._loss_func(
            unbinned_model,
            unbinned_data,
            fit_range,
            None,  # Constraints handled at hybrid level
            unbinned_offset
        )

        # Combine losses
        total_nll = binned_nll + unbinned_nll

        # Add constraints if present
        if constraints:
            constraints = z.reduce_sum([c.value() for c in constraints])
            total_nll += constraints

        return total_nll

    def create_new(
        self,
        binned_model: ztyping.BinnedPDFInputType = NONE,
        binned_data: ztyping.BinnedDataInputType = NONE,
        unbinned_model: ztyping.PDFInputType = NONE,
        unbinned_data: ztyping.DataInputType = NONE,
        fit_range=NONE,
        constraints=NONE,
        options=NONE,
    ):
        """Create a new HybridLoss with updated arguments."""
        if binned_model is NONE:
            binned_model = self._binned_loss.model
        if binned_data is NONE:
            binned_data = self._binned_loss.data
        if unbinned_model is NONE:
            unbinned_model = self._unbinned_loss.model
        if unbinned_data is NONE:
            unbinned_data = self._unbinned_loss.data
        if fit_range is NONE:
            fit_range = self.fit_range
        if constraints is NONE:
            constraints = self.constraints
        if options is NONE:
            options = self._options

        return type(self)(
            binned_model=binned_model,
            binned_data=binned_data,
            unbinned_model=unbinned_model,
            unbinned_data=unbinned_data,
            fit_range=fit_range,
            constraints=constraints,
            options=options,
        )