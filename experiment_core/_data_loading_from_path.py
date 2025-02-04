from pathlib import Path
from typing import Literal

import depth_tools
import voit
import voit.serialized

from ._load_paths import (
    get_hypersim_with_objects_inserted_subpath,
    get_nyuv2_with_objects_inserted_subpath,
    get_path_data,
)


def load_nyuv2_dataset(
    add_black_frame: bool,
    split: Literal["train", "test"],
) -> tuple[depth_tools.Nyuv2Dataset, depth_tools.DepthClip]:
    depth_clip = depth_tools.DepthClip((1e-3, 10))

    return (
        depth_tools.Nyuv2Dataset(
            dataset_dir=get_path_data().nyuv2,
            add_black_frame=add_black_frame,
            split=split,
        )
    ), depth_clip


def load_hypersim_dataset(
    split: Literal["train", "test", "val"], tonemap: Literal["jpeg", "reinhard"]
) -> tuple[depth_tools.SimplifiedHypersimDataset, depth_tools.DepthClip]:
    hypersim_dir = get_path_data().hypersim
    depth_clip = depth_tools.DepthClip((1e-3, 10))
    return (
        depth_tools.SimplifiedHypersimDataset(
            hypersim_dir=hypersim_dir, split=split, tonemap=tonemap
        ),
        depth_clip,
    )


def load_modified_hypersim_dataset(
    split: Literal["train", "test", "val"], tonemap: Literal["jpeg", "reinhard"]
) -> tuple[
    depth_tools.SimplifiedHypersimDataset,
    depth_tools.DepthClip,
    voit.serialized.DatasetWithObjectsInserted,
]:
    cache_dir = get_hypersim_with_objects_inserted_subpath(split=split, tonemap=tonemap)

    hypersim_dataset, depth_clip = load_hypersim_dataset(split=split, tonemap=tonemap)
    modified_dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=cache_dir)

    return hypersim_dataset, depth_clip, modified_dataset


def load_modified_nyuv2_dataset(split: Literal["train", "test"]) -> tuple[
    depth_tools.Nyuv2Dataset,
    depth_tools.DepthClip,
    voit.serialized.DatasetWithObjectsInserted,
]:
    cache_dir = get_nyuv2_with_objects_inserted_subpath(split=split)

    nyuv2_dataset, depth_clip = load_nyuv2_dataset(split=split, add_black_frame=True)
    modified_dataset = voit.serialized.DatasetWithObjectsInserted(cache_dir=cache_dir)

    return nyuv2_dataset, depth_clip, modified_dataset


def get_insertable_obj_keys() -> dict[str, Path]:
    """
    Get the key dictionary for the insertable objects.
    """
    result: dict[str, Path] = dict()
    glb_paths = get_path_data().insertable_objects.rglob("*.glb")
    for glb_path in glb_paths:
        key = f"{glb_path.parent.name}/{glb_path.name}"
        result[key] = glb_path

    return result
