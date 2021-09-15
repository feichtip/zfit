#  Copyright (c) 2021 zfit
import pydantic
import tensorflow as tf
import tensorflow_addons as tfa

from zfit.core.binning import unbinned_to_binindex
from zfit.core.interfaces import ZfitSpace
from zfit.core.space import supports
from zfit.models.functor import BaseFunctor
from zfit.z import numpy as znp


class UnbinnedFromBinnedPDF(BaseFunctor):

    def __init__(self, pdf, obs=None):
        if pdf.is_extended:
            extended = pdf.get_yield()
        if obs is None:
            obs = pdf.space
            obs = obs.with_binning(None)
        super().__init__(pdfs=pdf, obs=obs, extended=extended)
        self._binned_space = self.pdfs[0].space.with_obs(self.space)

    @supports(norm=True, multiple_limits=True)
    def _pdf(self, x, norm_range):
        binned_space = self.pdfs[0].space
        binindices = unbinned_to_binindex(x, binned_space, flow=True)
        pdf = self.pdfs[0]

        values = pdf.pdf(binned_space, norm=norm_range)

        # because we have the flow, so we need to make it here with pads
        padded_values = znp.pad(values, znp.ones((values.ndim, 2)), mode="constant")  # for overflow
        ordered_values = tf.gather_nd(padded_values, indices=binindices)
        return ordered_values

    @supports(norm=True, multiple_limits=True)
    def _ext_pdf(self, x, norm_range):
        binned_space = binned_space = self.pdfs[0].space
        binindices = unbinned_to_binindex(x, binned_space, flow=True)

        pdf = self.pdfs[0]

        values = pdf.ext_pdf(binned_space, norm=norm_range)
        ndim = len(values.shape)

        # because we have the flow, so we need to make it here with pads
        padded_values = znp.pad(values, znp.ones((ndim, 2)), mode="constant")  # for overflow
        ordered_values = tf.gather_nd(padded_values, indices=binindices)
        return ordered_values

    @supports(norm=True, multiple_limits=True)
    def _integrate(self, limits, norm_range):
        return self.pdfs[0].integrate(limits, norm=norm_range)

    @supports(norm=True, multiple_limits=True)
    def _ext_integrate(self, limits, norm_range):
        return self.pdfs[0].ext_integrate(limits, norm=norm_range)

    @supports(norm=True, multiple_limits=True)
    def _sample(self, n, limits: ZfitSpace):

        pdf = self.pdfs[0]
        # TODO: use real limits, currently not supported in binned sample
        sample = pdf.sample(n=n)

        edges = sample.space.binning.edges
        ndim = len(edges)
        edges = [znp.array(edge) for edge in edges]
        edges_flat = [znp.reshape(edge, [-1]) for edge in edges]
        lowers = [edge[:-1] for edge in edges_flat]
        uppers = [edge[1:] for edge in edges_flat]
        lowers_meshed = znp.meshgrid(*lowers, indexing='ij')
        uppers_meshed = znp.meshgrid(*uppers, indexing='ij')
        shape = tf.shape(lowers_meshed[0])
        lowers_meshed_flat = [znp.reshape(lower_mesh, [-1]) for lower_mesh in lowers_meshed]
        uppers_meshed_flat = [znp.reshape(upper_mesh, [-1]) for upper_mesh in uppers_meshed]
        lower_flat = znp.stack(lowers_meshed_flat, axis=-1)
        upper_flat = znp.stack(uppers_meshed_flat, axis=-1)

        counts_flat = znp.reshape(sample.values(), (-1,))
        lower_flat_repeated = tf.repeat(lower_flat, counts_flat, axis=0)
        upper_flat_repeated = tf.repeat(upper_flat, counts_flat, axis=0)
        sample_unbinned = tf.random.uniform((znp.sum(counts_flat), ndim),
                                            minval=lower_flat_repeated,
                                            maxval=upper_flat_repeated,
                                            dtype=self.dtype
                                            )
        return sample_unbinned


class SplinePDF(BaseFunctor):

    def __init__(self, pdf, order: int = None, obs=None,
                 extended=None):  # TODO: obs should not be needed? Or should it?
        if pdf.is_extended:
            extended = pdf.get_yield()
        if obs is None:
            obs = pdf.space
            obs = obs.with_binning(None)
        super().__init__(pdfs=pdf, obs=obs, extended=extended)
        if order is None:
            order = 3
        self._order = order

    @property
    def order(self):
        return self._order

    @supports(norm=True)
    def _ext_pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        density = pdf.ext_pdf(x.space, norm=norm_range)
        density_flat = znp.reshape(density, (-1,))
        centers_list = znp.meshgrid(*pdf.space.binning.centers, indexing='ij')
        centers_list_flat = [znp.reshape(cent, (-1,)) for cent in centers_list]
        centers = znp.stack(centers_list_flat, axis=-1)
        # [None, :, None]  # TODO: only 1 dim now
        probs = tfa.image.interpolate_spline(
            train_points=centers[None, ...],
            train_values=density_flat[None, :, None],
            query_points=x.value()[None, ...],
            order=self.order,

        )
        return probs[0, ..., 0]

    @supports(norm=True)
    def _pdf(self, x, norm_range):
        pdf = self.pdfs[0]
        density = pdf.pdf(x.space, norm=norm_range)  # TODO: order? Give obs, pdf makes order and binning herself?
        centers = pdf.space.binning.centers[0][None, :, None]  # TODO: only 1 dim now
        probs = tfa.image.interpolate_spline(
            train_points=centers,
            train_values=density[None, :, None],
            query_points=x.value()[None, ...],
            order=3

        )
        return probs[0, ..., 0]


class TypedSplinePDF(pydantic.BaseModel):
    order: pydantic.conint(ge=0)
