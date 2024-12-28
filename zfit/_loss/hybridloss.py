#  Copyright (c) 2024 zfit

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal, Optional, Union

import pydantic.v1 as pydantic
from pydantic.v1 import Field

from .. import z
from .._loss.binnedloss import (_check_small_counts_chi2, chi2_loss_calc,
                                poisson_loss_calc)
from ..core.data import convert_to_data
from ..core.interfaces import (ZfitBinnedData, ZfitBinnedPDF, ZfitData,
                               ZfitParameter, ZfitPDF)
from ..core.loss import BaseLoss, _unbinned_nll_tf
from ..exception import OutsideLimitsError
from ..serialization.serializer import BaseRepr, Serializer
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container
from ..util.exception import IntentionAmbiguousError, NotExtendedPDFError
from ..util.warnings import warn_advanced_feature
from ..z import numpy as znp


class HybridLossRepr(BaseRepr):
    _implementation = None
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["HybridLoss"] = Field("HybridLoss", alias="type")
    model_binned: Union[
        Serializer.types.PDFTypeDiscriminated,
        list[Serializer.types.PDFTypeDiscriminated],
    ]
    model_unbinned: Union[
        Serializer.types.PDFTypeDiscriminated,
        list[Serializer.types.PDFTypeDiscriminated],
    ]
    data_binned: Union[
        Serializer.types.DataTypeDiscriminated,
        list[Serializer.types.DataTypeDiscriminated],
    ]
    data_unbinned: Union[
        Serializer.types.DataTypeDiscriminated,
        list[Serializer.types.DataTypeDiscriminated],
    ]
    constraints: Optional[list[Serializer.types.ConstraintTypeDiscriminated]] = Field(default_factory=list)
    options: Optional[dict] = Field(default_factory=dict)


class BaseHybridLoss(BaseLoss):
    """Base class for hybrid losses combining binned and unbinned components."""

    def __init__(
        self,
        model_binned: ztyping.BinnedPDFInputType,
        model_unbinned: ztyping.PDFInputType,
        data_binned: ztyping.BinnedDataInputType,
        data_unbinned: ztyping.DataInputType,
        constraints: ztyping.ConstraintsInputType = None,
        options: dict | None = None,
    ):
        # Convert inputs to containers
        model_binned = convert_to_container(model_binned)
        model_unbinned = convert_to_container(model_unbinned)
        data_binned = convert_to_container(data_binned)
        data_unbinned = convert_to_container(data_unbinned)

        # Input validation
        if not len(model_binned) == len(data_binned):
            msg = (
                "Number of binned models and datasets must match. "
                f"Got {len(model_binned)} models and {len(data_binned)} datasets."
            )
            raise ValueError(msg)

        if not len(model_unbinned) == len(data_unbinned):
            msg = (
                "Number of unbinned models and datasets must match. "
                f"Got {len(model_unbinned)} models and {len(data_unbinned)} datasets."
            )
            raise ValueError(msg)

        # Check types
        not_binned_pdf = [mod for mod in model_binned if not isinstance(mod, ZfitBinnedPDF)]
        not_binned_data = [dat for dat in data_binned if not isinstance(dat, ZfitBinnedData)]

        error_msg = ""
        if not_binned_pdf:
            error_msg += (
                "The following PDFs are not binned but need to be. They can be wrapped in a "
                f"BinnedFromUnbinnedPDF: {not_binned_pdf}"
            )
        if not_binned_data:
            error_msg += (
                "The following datasets are not binned but need to be. They can be converted using "
                f"the `to_binned` method: {not_binned_data}"
            )
        if error_msg:
            raise ValueError(error_msg)

        # Handle unbinned data conversion if needed
        data_unbinned_checked = []
        for mod, dat in zip(model_unbinned, data_unbinned):
            if not isinstance(dat, ZfitData):
                try:
                    dat = convert_to_data(data=dat, obs=mod.space, check_limits=True)
                except OutsideLimitsError as error:
                    msg = (
                        f"Data {dat} is not a zfit Data and is not fully within the limits {mod.space} "
                        f"of the model {mod}. Please convert to zfit Data or remove events outside the space."
                    )
                    raise IntentionAmbiguousError(msg) from error
            data_unbinned_checked.append(dat)

        # Combine all models and data for BaseLoss initialization
        all_models = [*model_binned, *model_unbinned]
        all_data = [*data_binned, *data_unbinned_checked]

        super().__init__(
            model=all_models,
            data=all_data,
            constraints=constraints,
            fit_range=None,
            options=options,
        )

        # Store components separately
        self._model_binned = model_binned
        self._model_unbinned = model_unbinned
        self._data_binned = data_binned
        self._data_unbinned = data_unbinned_checked
        self._errordef = 0.5

    @property
    def model_binned(self):
        """Return the binned model components."""
        return self._model_binned

    @property
    def model_unbinned(self):
        """Return the unbinned model components."""
        return self._model_unbinned

    @property
    def data_binned(self):
        """Return the binned data components."""
        return self._data_binned

    @property
    def data_unbinned(self):
        """Return the unbinned data components."""
        return self._data_unbinned


class HybridLoss(BaseHybridLoss):
    _name = "HybridLoss"

    def __init__(
        self,
        model_binned: ztyping.BinnedPDFInputType,
        model_unbinned: ztyping.PDFInputType,
        data_binned: ztyping.BinnedDataInputType,
        data_unbinned: ztyping.DataInputType,
        constraints: ztyping.ConstraintsInputType = None,
        options: dict | None = None,
    ):
        r"""Combined binned and unbinned negative log likelihood loss.

        This loss function combines binned and unbinned negative log likelihoods into a single loss
        with equal weights:

        .. math::
            \mathcal{L}_{hybrid} = \mathcal{L}_{binned} + \mathcal{L}_{unbinned}

        The binned and unbinned components contribute equally to the total likelihood, allowing
        simultaneous fitting of both types of data.

        Args:
            model_binned: Binned PDF(s) that return the normalized probability for binned data
            model_unbinned: Unbinned PDF(s) that return the normalized probability for unbinned data
            data_binned: Binned dataset(s) that will be given to the binned model(s)
            data_unbinned: Unbinned dataset(s) that will be given to the unbinned model(s)
            constraints: Auxiliary measurements ("constraints") that add a likelihood term
            options: Additional options for the loss calculation
        """
        super().__init__(
            model_binned=model_binned,
            model_unbinned=model_unbinned,
            data_binned=data_binned,
            data_unbinned=data_unbinned,
            constraints=constraints,
            options=options,
        )

        # Check for extended PDFs and warn
        extended_pdfs = [pdf for pdf in model_binned + model_unbinned if pdf.is_extended]
        if extended_pdfs and type(self) is HybridLoss:
            warn_advanced_feature(
                f"Extended PDFs ({extended_pdfs}) will be treated as non-extended in HybridLoss. "
                "Use ExtendedHybridLoss for proper handling of yields.",
                identifier="extended_in_HybridLoss"
            )

    @z.function(wraps="loss")
    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        """Calculate the combined loss from binned and unbinned components."""
        # Split models and data back into binned and unbinned components
        n_binned = len(self._model_binned)
        model_binned = model[:n_binned]
        model_unbinned = model[n_binned:]
        data_binned = data[:n_binned]
        data_unbinned = data[n_binned:]

        # Calculate binned NLL
        nll_binned = 0
        for mod, dat in zip(model_binned, data_binned):
            values = dat.values()
            variances = dat.variances()
            probs = mod.rel_counts(dat)
            probs *= znp.sum(values)
            poisson_term = poisson_loss_calc(probs, values, log_offset, variances)
            nll_binned += znp.sum(poisson_term)

        # Calculate unbinned NLL
        nll_unbinned, nll_corr = _unbinned_nll_tf(
            model=model_unbinned,
            data=data_unbinned,
            fit_range=None,
            log_offset=log_offset,
            kahan=self._options.get("sumtype") == "kahan"
        )

        # Add constraint terms if any
        total_nll = nll_binned + nll_unbinned - nll_corr
        if constraints:
            constraints_sum = z.reduce_sum([c.value() for c in constraints])
            total_nll += constraints_sum

        return total_nll

    @property
    def is_extended(self):
        """Whether the loss takes yields into account."""
        return False

    def _get_params(
        self,
        floating: bool | None,
        is_yield: bool | None,
        extract_independent: bool | None,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        if not self.is_extended:
            is_yield = False  # Loss doesn't depend on yields
        return super()._get_params(floating, is_yield, extract_independent, autograd=autograd)

    def create_new(
        self,
        model_binned: ztyping.BinnedPDFInputType = NONE,
        model_unbinned: ztyping.PDFInputType = NONE,
        data_binned: ztyping.BinnedDataInputType = NONE,
        data_unbinned: ztyping.DataInputType = NONE,
        constraints: ztyping.ConstraintsInputType = NONE,
        options: dict | None = NONE,
    ):
        """Create a new hybrid loss with optionally updated components."""
        if model_binned is NONE:
            model_binned = self.model_binned
        if model_unbinned is NONE:
            model_unbinned = self.model_unbinned
        if data_binned is NONE:
            data_binned = self.data_binned
        if data_unbinned is NONE:
            data_unbinned = self.data_unbinned
        if constraints is NONE:
            constraints = self.constraints
            if constraints is not None:
                constraints = constraints.copy()
        if options is NONE:
            options = self._options
            if isinstance(options, dict):
                options = options.copy()

        return type(self)(
            model_binned=model_binned,
            model_unbinned=model_unbinned,
            data_binned=data_binned,
            data_unbinned=data_unbinned,
            constraints=constraints,
            options=options,
        )


class ExtendedHybridLoss(BaseHybridLoss):
    _name = "ExtendedHybridLoss"

    def __init__(
        self,
        model_binned: ztyping.BinnedPDFInputType,
        model_unbinned: ztyping.PDFInputType,
        data_binned: ztyping.BinnedDataInputType,
        data_unbinned: ztyping.DataInputType,
        constraints: ztyping.ConstraintsInputType = None,
        options: dict | None = None,
    ):
        r"""Extended hybrid loss combining binned and unbinned components with yield terms.

        This loss function combines extended binned and unbinned negative log likelihoods:

        .. math::
            \mathcal{L}_{hybrid} = \mathcal{L}_{binned} + \mathcal{L}_{unbinned} +
            \sum_i \mathcal{L}_{yield,i}

        The yield terms account for the total number of events in each component.

        Args:
            model_binned: Extended binned PDF(s) for binned data
            model_unbinned: Extended unbinned PDF(s) for unbinned data
            data_binned: Binned dataset(s)
            data_unbinned: Unbinned dataset(s)
            constraints: Optional auxiliary measurement constraints
            options: Additional options for loss calculation
        """
        super().__init__(
            model_binned=model_binned,
            model_unbinned=model_unbinned,
            data_binned=data_binned,
            data_unbinned=data_unbinned,
            constraints=constraints,
            options=options,
        )

        # Verify all PDFs are extended
        not_extended = []
        for mod in model_binned + model_unbinned:
            if not mod.is_extended:
                not_extended.append(mod)
        if not_extended:
            msg = f"The following PDFs are not extended but must be for ExtendedHybridLoss: {not_extended}"
            raise NotExtendedPDFError(msg)

    @z.function(wraps="loss")
    def _loss_func(self, model, data, fit_range, constraints, log_offset):
        """Calculate the combined extended loss."""
        # Split models and data
        n_binned = len(self._model_binned)
        model_binned = model[:n_binned]
        model_unbinned = model[n_binned:]
        data_binned = data[:n_binned]
        data_unbinned = data[n_binned:]

        # Calculate binned NLL with yields
        nll_binned = 0
        for mod, dat in zip(model_binned, data_binned):
            values = dat.values()
            variances = dat.variances()
            probs = mod.counts(dat)  # Use counts instead of rel_counts for extended
            poisson_term = poisson_loss_calc(probs, values, log_offset, variances)
            nll_binned += znp.sum(poisson_term)

        # Calculate unbinned NLL
        nll_unbinned, nll_corr = _unbinned_nll_tf(
            model=model_unbinned,
            data=data_unbinned,
            fit_range=None,
            log_offset=log_offset,
            kahan=self._options.get("sumtype") == "kahan"
        )

        # Add yield terms for both components
        yields = []
        nevents_collected = []

        # Collect yields and event counts from both binned and unbinned components
        for mod, dat in zip(model_binned + model_unbinned, data_binned + data_unbinned):
            nevents = dat.n_events if dat.weights is None else z.reduce_sum(dat.weights)
            nevents = znp.asarray(nevents, tf.float64)
            nevents_collected.append(nevents)
            yields.append(znp.atleast_1d(mod.get_yield()))

        yields = znp.concatenate(yields, axis=0)
        nevents_collected = znp.stack(nevents_collected, axis=0)

        # Calculate Poisson term for yields
        term_new = tf.nn.log_poisson_loss(
            nevents_collected,
            znp.log(yields),
            compute_full_loss=log_offset is False
        )

        if log_offset is not False:
            log_offset = znp.asarray(log_offset, dtype=znp.float64)
            term_new += log_offset

        nll_yields = znp.sum(term_new, axis=0)

        # Combine all terms
        total_nll = nll_binned + nll_unbinned - nll_corr + nll_yields

        # Add constraints if any
        if constraints:
            constraints_sum = z.reduce_sum([c.value() for c in constraints])
            total_nll += constraints_sum

        return total_nll

    @property
    def is_extended(self):
        """Whether the loss takes yields into account."""
        return True

    def create_new(
        self,
        model_binned: ztyping.BinnedPDFInputType = NONE,
        model_unbinned: ztyping.PDFInputType = NONE,
        data_binned: ztyping.BinnedDataInputType = NONE,
        data_unbinned: ztyping.DataInputType = NONE,
        constraints: ztyping.ConstraintsInputType = NONE,
        options: dict | None = NONE,
    ):
        """Create a new extended hybrid loss with optionally updated components."""
        if model_binned is NONE:
            model_binned = self.model_binned
        if model_unbinned is NONE:
            model_unbinned = self.model_unbinned
        if data_binned is NONE:
            data_binned = self.data_binned
        if data_unbinned is NONE:
            data_unbinned = self.data_unbinned
        if constraints is NONE:
            constraints = self.constraints
            if constraints is not None:
                constraints = constraints.copy()
        if options is NONE:
            options = self._options
            if isinstance(options, dict):
                options = options.copy()

        return type(self)(
            model_binned=model_binned,
            model_unbinned=model_unbinned,
            data_binned=data_binned,
            data_unbinned=data_unbinned,
            constraints=constraints,
            options=options,
        )
