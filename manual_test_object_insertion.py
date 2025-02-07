import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Sequence

import depth_tools
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import experiment_core


def main():
    print("Loading the dataset")
    original_dataset, _, modified_dataset = experiment_core.load_modified_nyuv2_dataset(
        split="train"
    )

    print(
        "Verify whether the given sample correctly appears in both 2D and 3D and the original sample and the sample with the object inserted are consistent."
    )
    fig, axs = plt.subplots(nrows=2, ncols=2)

    modified_sample_0 = modified_dataset[0]
    original_sample_0 = original_dataset[modified_dataset.original_indices[0]]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        html_path = td / "depth.html"
        depth_tools.depths_2_plotly_fig(
            depth_maps=[
                {
                    "color": modified_sample_0["rgb"],
                    "depth_map": modified_sample_0["depth"],
                    "depth_mask": modified_sample_0["depth"] < 10,
                    "name": "Sample0",
                    "size": 1,
                }
            ],
            subsample={
                "max_num": 30000,
            },
            coord_sys=depth_tools.CoordSys.LH_YUp,
            intrinsics=modified_sample_0["camera"],
            title="Depth map",
        ).write_html(html_path)
        webbrowser.open(str(html_path))
        time.sleep(10)

    axs[0, 0].set_title("Original RGB")
    axs[0, 0].imshow(original_sample_0["rgb"].transpose([1, 2, 0]))
    axs[0, 1].set_title("Original depth")
    axs[0, 1].imshow(original_sample_0["depth"][0])
    axs[1, 0].set_title("Modified RGB")
    axs[1, 0].imshow(modified_sample_0["rgb"].transpose([1, 2, 0]))
    axs[1, 1].set_title("Modified depth")
    axs[1, 1].imshow(modified_sample_0["depth"][0])
    plt.show(block=True)


if __name__ == "__main__":
    main()
