import shutil

import torch
import voit.serialized

import experiment_core


def main():
    nyuv2_train = experiment_core.load_nyuv2_dataset(
        add_black_frame=True, split="train"
    )[0]
    sample0 = nyuv2_train[0]

    global_im_size = voit.Vec2i(
        sample0["rgb"].shape[2],
        sample0["rgb"].shape[1],
    )

    inserter = voit.serialized.DatasetBasedInserter(
        original_samples={
            "dataset": nyuv2_train,
            "global_camera": nyuv2_train[0]["camera"],
            "global_im_size": global_im_size,
            "is_depth_usable_for_depth_maps": True,
            "is_im_linear": False,
        },
        floor_proxy_size=voit.Vec2(5, 5),
        objs_and_keys=experiment_core.get_insertable_obj_keys(),
        pt_device=torch.device("cuda"),
        output_im_linear=False,
    )
    insertion_specs = voit.serialized.load_insertion_specs(
        experiment_core.get_nyuv2_insertion_specs_path("train")
    )

    insertion_cache_dir = experiment_core.get_nyuv2_with_objects_inserted_subpath(
        "train"
    )

    if insertion_cache_dir.exists():
        shutil.rmtree(insertion_cache_dir)

    insertion_cache_dir.mkdir(parents=True)

    voit.serialized.DatasetWithObjectsInserted.generate(
        inserter=inserter,
        cache_dir=experiment_core.get_nyuv2_with_objects_inserted_subpath("train"),
        insertion_specs=insertion_specs,
        report_progress_tqdm=True,
    )


if __name__ == "__main__":
    main()
