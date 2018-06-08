import argparse
from functools import partial
from numbers import Number
from typing import Union, Tuple

import numpy as np
from skimage import img_as_uint
from skimage.filters import gaussian

from starfish.errors import DataFormatWarning
from starfish.io import Stack
from ._base import FilterAlgorithmBase


# TODO ambrosejcarr: need a better solution for 2d/3d support for image analysis algorithms

class GaussianLowPass(FilterAlgorithmBase):

    def __init__(
            self, sigma: Union[Number, Tuple[Number]], is_volume: bool=False, verbose: bool=False, **kwargs
    ) -> None:
        """Multi-dimensional low-pass gaussian filter.

        Parameters
        ----------
        sigma : Union[Number, Tuple[Number]]
            Standard deviation for Gaussian kernel.
        is_volume : bool
            If True, 3d (z, y, x) volumes will be filtered, otherwise, filter 2d tiles independently.
        verbose : bool
            If True, report on the percentage completed (default = False) during processing

        """
        if isinstance(sigma, tuple):
            message = ("if passing an anisotropic kernel, the dimensionality must match the data shape ({shape}), not "
                       "{passed_shape}")
            if is_volume and len(sigma) != 3:
                raise ValueError(message.format(shape=3, passed_shape=len(sigma)))
            if not is_volume and len(sigma) != 2:
                raise ValueError(message.format(shape=2, passed_shape=len(sigma)))

        self.sigma = sigma
        self.is_volume = is_volume
        self.verbose = verbose

    @classmethod
    def get_algorithm_name(cls) -> str:
        return "gaussian_low_pass"

    @classmethod
    def add_arguments(cls, group_parser: argparse.ArgumentParser) -> None:
        group_parser.add_argument(
            "--sigma", type=float, help="standard deviation of gaussian kernel")
        group_parser.add_argument(
            "--is-volume", action="store_true", help="indicates that the image stack should be filtered in 3d")

    @staticmethod
    def low_pass(image: np.ndarray, sigma: Union[Number, Tuple[Number]]) -> np.ndarray:
        """
        Apply a Gaussian blur operation over a multi-dimensional image.

        Parameters
        ----------
        image : np.ndarray[np.uint32]
            2-d or 3-d image data
        sigma : Union[Number, Tuple[Number]]
            Standard deviation of the Gaussian kernel that will be applied. If a float, an isotropic kernel will be
            assumed, otherwise the dimensions of the kernel give (z, y, x)

        Returns
        -------
        np.ndarray :
            Blurred data in same shape as input image, converted to np.uint16 dtype.

        """
        if image.dtype != np.uint16:
            DataFormatWarning('gaussian filters only support uint16 images. Image data will be converted')
            image = img_as_uint(image)

        blurred = gaussian(
            image, sigma=sigma, output=None, cval=0, multichannel=True, preserve_range=True, truncate=4.0)

        blurred = blurred.clip(0).astype(np.uint16)

        return blurred

    def filter(self, stack: Stack) -> None:
        """
        Perform in-place filtering of an image stack and all contained aux images.

        Parameters
        ----------
        stack : starfish.Stack
            Stack to be filtered.

        """
        low_pass = partial(self.low_pass, sigma=self.sigma)
        stack.image.apply(low_pass, is_volume=self.is_volume, verbose=self.verbose)

        # apply to aux dict too:
        for auxiliary_image in stack.auxiliary_images.values():
            auxiliary_image.apply(low_pass, is_volume=self.is_volume)
