import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jsonschema

_path_data = None

_PATHS_SCHEMA = {
    "type": "object",
    "properties": {"hypersim": {"type": "string"}, "nyuv2": {"type": "string"}},
    "required": ["hypersim", "nyuv2"],
}


@dataclass(frozen=True)
class PathData:
    hypersim: Path
    nyuv2: Path
    splits_dir: Path
    object_insertion_cache: Path
    insertable_objects: Path
    cache_dirs_root: Path
    insertion_specs: Path


def get_path_data() -> PathData:
    global _path_data

    if _path_data is not None:
        return _path_data

    project_root_dir = Path(__file__).parent.parent

    paths_json_path = project_root_dir / "paths.json"

    if not paths_json_path.is_file():
        paths_json_default_path = paths_json_path.with_name("paths_default.json")
        shutil.copy(src=paths_json_default_path, dst=paths_json_path)
        logging.warning(
            f'The paths file "{paths_json_path}" was not found. Copying the default path data file from "{paths_json_default_path}".'
        )

    paths_json = json.loads(paths_json_path.read_text())
    jsonschema.validate(instance=paths_json, schema=_PATHS_SCHEMA)

    splits_dir = project_root_dir / "splits"

    insertion_specs_dir = project_root_dir / "insertion_specs"

    _path_data = PathData(
        hypersim=Path(paths_json["hypersim"]),
        splits_dir=splits_dir,
        nyuv2=Path(paths_json["nyuv2"]),
        object_insertion_cache=Path(paths_json["object_insertion_cache"]),
        insertable_objects=Path(paths_json["insertable_objects"]),
        cache_dirs_root=Path(paths_json["cache_dirs_root"]),
        insertion_specs=insertion_specs_dir,
    )

    return _path_data


def get_hypersim_with_objects_inserted_subpath(
    split: Literal["train", "test", "val"], tonemap: Literal["jpeg", "reinhard"]
) -> Path:
    return get_path_data().object_insertion_cache / f"hypersim_{split}_{tonemap}"


def get_nyuv2_with_objects_inserted_subpath(split: Literal["train", "test"]) -> Path:
    return get_path_data().object_insertion_cache / f"nyuv2_{split}"


def get_nyuv2_insertion_specs_path(split: Literal["train", "test"]) -> Path:
    return get_path_data().insertion_specs / f"nyuv2_{split}_realistic_samples.json"


def get_notebook_cache_dir(notebook_id: str, cache_id: str) -> Path:
    if not re.match(r"^[a-zA-Z0-9_\-]+$", notebook_id):
        raise ValueError(
            Rf'The notebobok id {notebook_id} does not match to regular expression "^[a-zA-Z0-9_\-]+$".'
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", cache_id):
        raise ValueError(
            Rf'The cache id {cache_id} does not match to regular expression "^[a-zA-Z0-9_\-]+$".'
        )

    return get_path_data().cache_dirs_root / notebook_id / cache_id
