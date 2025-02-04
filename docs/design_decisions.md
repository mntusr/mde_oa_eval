rejected hypersim scenes

* everything in issue 22
  * reason: the developers recognized that these scenes should be ignored
* the scenes to reject described at `evermotion_dataset/analysis/metadata_images_split_scene_v1.csv`
  * reason: the developers decided to reject these scenes
* the scenes that use the VRay-specific tilt-shift parameters
  * core idea of these parameters:
    * the image plane is not orthogonal to the far-axis
      * concise explanation: <https://github.com/apple/ml-hypersim/issues/24#issuecomment-944430022>
  * issues:
    * <https://github.com/apple/ml-hypersim/issues/76>
    * <https://github.com/apple/ml-hypersim/issues/24>
    * <https://github.com/apple/ml-hypersim/issues/9>
  * ⇓
  * reasons for decision
    * these tilt-shift parameters are not relevant for industrial applications, they are more artistic
    * they need extensive modifications in our implementation
    * the concept of depth map is ambiguous for such cameras