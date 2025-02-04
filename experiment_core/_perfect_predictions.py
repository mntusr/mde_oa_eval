from typing import Final, SupportsIndex

import depth_tools
import numpy as np


class AlmostPerfectPredictions:
    """
    A class that emulates a depth prediction model with controllable error.

    Parameters
    ----------
    dataset
        The dataset that will be used for evaluation.
    error
        A variable that controls the amount of error.
    """

    def __init__(self, dataset: depth_tools.Dataset, error: float) -> None:
        self.dataset = dataset
        self.error: Final = error

    def __getitem__(self, idx: SupportsIndex) -> np.ndarray:
        depth = self.dataset[idx]["depth"].copy()

        depth = (depth - depth.mean()) * (self.error + 1) + depth.mean()

        return depth
