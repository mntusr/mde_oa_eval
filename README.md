Required data

- [Hypersim dataset](https://github.com/apple/ml-hypersim/). You need the following data:
  - `*.depth_meters.hdf5`
  - `*.tonemap.jpg`
- [NYU Depth v2 dataset](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html). You need the following data:
  - Labeled dataset
  - [Eigen splits](http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat)
- The directory of the 3D inserted 3d objects
  - [Download from the release](https://github.com/mntusr/mde_oa_eval/releases/download/v1.0.0/inserted_assets.zip)

Required environment:

- `uv` installed

Steps to run the measurements

1. Download the required data and clone/download this repository.
2. If you do not use CUDA, modify `pyproject.toml` accordingly.
3. Run `uv sync`
4. Copy the `paths_default.json` file to `paths.json`.
5. Modify the paths to fit your needs:
   - `hypersim`: The path of the downloaded and extracted Hypersim dataset.
   - `nyuv2`: The path of the downloaded Nyu Depth v2 dataset.
   - `object_insertion_cache`: Path of the cache that stores the results of the object insertion tool.
   - `insertable_objects`: The path of the insertable objects.
   - `cache_dirs_root`: The path of the directory that stores all cache except object the object insertion cache.
6. Run the unit tests to verify if everything works as intended.
7. Run `manual_test_model.py` to verify that the model works as expected. Follow the instructions of the test. Follow the instructions of the test.
8. Run `generate_insertion_cache.py` to generate the object insertion cache.
9. Run `manual_test_object_insertion.py` to verify whether the object insertion works correctly.
