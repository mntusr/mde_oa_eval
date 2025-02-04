import shutil
from collections.abc import Collection
from enum import Enum, auto
from pathlib import Path
from typing import Final, Iterable, Protocol, Sequence, SupportsIndex, cast

import depth_tools
import numpy as np
import PIL.Image as Image
import torch
import transformers
import transformers.models
from tqdm import tqdm, trange


class MetricDepthPredictor(Protocol):
    def predict_metric_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Do metric depth prediction.

        Parameters
        ----------
        image
            The input image. Format: ``Im_RGB``

        Returns
        -------
        v
            The predicted metric depth map. It has the same size as the input image. Format: ``Im_Depth``
        """
        ...


SUPPORTED_MODELS: Final[tuple[str, ...]] = ("dummy", "depth_anything_v2", "zoe_depth")


def load_model(predictor: str, device: str = "cpu") -> MetricDepthPredictor:
    match predictor:
        case "depth_anything_v2":
            return DepthAnythingV2(device=device)
        case "zoe_depth":
            return ZoeDepth(device=device)
        case "dummy":
            return Dummy(device=device)
        case _:
            raise ValueError(f'Unknown model "{predictor}"')


# TODO test DepthAnythingV2
class DepthAnythingV2:
    def __init__(self, device: str = "cpu"):
        self._pipe = transformers.pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )

    def predict_metric_depth(self, image: np.ndarray) -> np.ndarray:
        image = (image * 255).astype(np.uint8).transpose([1, 2, 0])
        image_im = Image.fromarray(image)
        depth: Image.Image = self._pipe(image_im)["depth"]  # type: ignore
        depth_np = np.array(depth)
        return depth_np


class Dummy:
    def __init__(self, device: str = "cpu"):
        self._pipe = transformers.pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device,
        )

    def predict_metric_depth(self, image: np.ndarray) -> np.ndarray:
        depth_np = np.ones((1, image.shape[1], image.shape[2]), dtype=image.dtype)
        return depth_np


class ZoeDepth:
    """
    A depth predictor based on the Huggingface version of ZoeDepth-M12-NK.

    This model variant uses a classifier to detect the proper environment and configure the maximal depth accordingly.

    Parameters
    ----------
    device
        The device on which the depth predictor runs.
    """

    def __init__(self, device: str = "cpu"):
        self._image_processor = cast(
            transformers.ZoeDepthImageProcessor,
            transformers.ZoeDepthImageProcessor.from_pretrained(
                "Intel/zoedepth-nyu-kitti"
            ),
        )
        self._model = cast(
            transformers.ZoeDepthForDepthEstimation,
            transformers.ZoeDepthForDepthEstimation.from_pretrained(
                "Intel/zoedepth-nyu-kitti"
            ),
        )
        self._model.to(device)  # type: ignore
        self._model.eval()
        self.device: Final = device

    def predict_metric_depth(self, image: np.ndarray) -> np.ndarray:
        image = (image * 255).astype(np.uint8).transpose([1, 2, 0])
        image_im = Image.fromarray(image)
        inputs = self._image_processor.preprocess(
            images=image_im, return_tensors="pt", do_resize=False, do_pad=False
        )

        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            zoe_outputs = self._model(**inputs)
            zoe_predicted_depth = zoe_outputs.predicted_depth.cpu().numpy()
        return zoe_predicted_depth


class DepthPredictionCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir: Final = cache_dir

    def __getitem__(self, idx: SupportsIndex) -> np.ndarray:
        return np.load(self.cache_dir / f"pred_{idx:06}.npy")

    @staticmethod
    def generate(
        *,
        cache_dir: Path,
        predictor: MetricDepthPredictor,
        dataset: depth_tools.Dataset,
        indices: Collection[int] | None = None,
    ) -> "DepthPredictionCache":
        if indices is None:
            indices = np.arange(len(dataset))  # TODO add tests

        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True)
        print("Generating prediction cache")
        for sample_idx in tqdm(indices):
            metric_depth = predictor.predict_metric_depth(dataset[sample_idx]["rgb"])
            np.save(cache_dir / f"pred_{sample_idx:06}.npy", metric_depth)

        return DepthPredictionCache(cache_dir)
