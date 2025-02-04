import numpy as np


def generate_idx_set(
    n_all_elements: int,
    n_elements_to_choose: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Generate an array that contains indices in a dataset.

    Parameters
    ----------
    n_all_elements
        The number of elements in the original dataset.
    n_elements_to_choose
        The number of elements to choose (without replacement).

    Returns
    --------
    v
        The array containing the chosen indices. Format: ``Indices``

    Raises
    ------
    ValueError
        If any of the element counts is non-positive.
    """
    if n_all_elements < n_elements_to_choose:
        raise ValueError(
            f"The number of elements ({n_all_elements}) to choose is greater than the total number of elements ({n_elements_to_choose}). In this case, the sampling is not possible without replacement."
        )

    if n_all_elements <= 0:
        raise ValueError(
            f"The number of elements in the dataset should be at least 1. Current value: {n_all_elements}"
        )

    if n_elements_to_choose <= 1:
        raise ValueError(
            f"The number of elements to choose should be at least 1. Current value: {n_elements_to_choose}"
        )

    if rng is None:
        rng = np.random.default_rng()
    idx_array = rng.choice(n_all_elements, n_elements_to_choose, replace=False)
    return [e.item() for e in idx_array]
