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
from ..core.interfaces import ZfitBinnedData, ZfitBinnedPDF, ZfitData, ZfitPDF
from ..core.loss import BaseLoss, _unbinned_nll_tf
from ..serialization.serializer import BaseRepr, Serializer
from ..util import ztyping
from ..util.checks import NONE
from ..util.container import convert_to_container
from ..util.exception import IntentionAmbiguousError
from ..exception import OutsideLimitsError
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


class HybridLoss(BaseLoss):
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

        This loss function combines binned and unbinned negative log likelihoods into a single loss:

        .. math::
            \mathcal{L}_{hybrid} = \mathcal{L}_{binned} * \mathcal{L}_{unbinned}

        Args:
            model_binned: Binned PDF(s) that return the normalized probability for binned data
            model_unbinned: Unbinned PDF(s) that return the normalized probability for unbinned data
            data_binned: Binned dataset(s) that will be given to the binned model(s)
            data_unbinned: Unbinned dataset(s) that will be given to the unbinned model(s)
            constraints: Auxiliary measurements ("constraints") that add a likelihood term
            options: Additional options for the loss calculation
        """
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
