import operator
from typing import Callable, Iterator, Literal, Protocol, SupportsIndex

import cv2
import depth_tools
import numpy as np
from matplotlib import pyplot as plt


class DepthDatasetWithError:
    """
    This class implements a depth dataset that simulates a Kinect-like depth blur atop an existing depth dataset.

    Parameters
    ----------
    dataset
        The atop which the blur should be simulated.
    blur_size
        The size of the kernel to calculate the blur.
    multiplicative_noise
        The function that generates the multiplicative noise to apply **after** the blur.

    Raises
    ------
    ValueError
        If the blur size is non-positive.
    """

    def __init__(
        self,
        *,
        dataset: depth_tools.Dataset,
        blur_size: int,
        multiplicative_noise_fn: Callable[[int], np.ndarray] | None = None,
    ) -> None:
        if blur_size <= 0:
            raise ValueError(f"The blur size ({blur_size}) is non-positive.")

        self.dataset = dataset

        self._kernel = np.ones((blur_size, blur_size))
        self._kernel_area = blur_size * blur_size

        self.multiplicative_noise_fn = multiplicative_noise_fn

    @property
    def blur_size(self) -> int:
        """
        The size of the calculated blur.
        """
        return self._kernel.shape[0]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: SupportsIndex, /) -> depth_tools.Sample:
        original_sample = self.dataset[idx]

        depth_with_zeros_at_not_mask = original_sample["depth"].copy()
        depth_with_zeros_at_not_mask[~self.dataset[idx]["mask"]] = 0

        depth_kernel_sums = self._apply_convolution(depth_with_zeros_at_not_mask)

        # plt.imshow(depth_kernel_sums[0])
        # plt.show(block=True)
        mask_kernel_sums = self._apply_convolution(
            original_sample["mask"].astype(np.float32)
        )

        blurred_depth = np.zeros_like(depth_kernel_sums)
        blurred_depth[mask_kernel_sums > 0] = (
            depth_kernel_sums[mask_kernel_sums > 0]
            / mask_kernel_sums[mask_kernel_sums > 0]
        )

        if self.multiplicative_noise_fn is not None:
            noise_vals = self.multiplicative_noise_fn(operator.index(idx))

            noise_vals = np.clip(noise_vals, 0, None)
            blurred_depth = blurred_depth * noise_vals

        return {
            "depth": blurred_depth,
            "mask": original_sample["mask"],
            "camera": original_sample["camera"],
            "name": original_sample["name"],
            "rgb": original_sample["rgb"],
        }

    def _apply_convolution(self, im: np.ndarray) -> np.ndarray:
        """
        Apply convolution on a scalar image.

        The outside pixels are treated as if they contained consant 0-s.

        The convolution kernel is given by the ``_kernel`` field.

        Parameters
        ----------
        im
            The scalar image on which the convolution is applied. Format: ``Im_Scalar``

        Returns
        -------
        v
            The result of the convolution. Format: ``Im_Scalar``
        """
        return np.expand_dims(
            cv2.filter2D(
                im[0],
                -1,
                self._kernel,
                borderType=cv2.BORDER_CONSTANT,
            ),
            axis=0,
        )

    def __iter__(self) -> Iterator[depth_tools.Sample]:
        for i in range(len(self)):
            yield self[i]
