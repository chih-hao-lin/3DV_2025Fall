### Downloading Necessary Files

```bash
./download_gdrive_data.sh
```

This will download the files as `assets` folder. It contains the following files:
- `assets/body_models/`
- `assets/robot_asset/`
- `assets/checkpoints/`
- `assets/ckpt_raft/`
- `assets/ckpt_sam2/`
- `assets/configs/`
- `assets/robot_asset/`

### Output Root

Prepare `demo_data` directory. It is recommend to be Symbolic link to a hard drive, since the output files are large.

```bash
ln -s /path/to/hard/drive/demo_data ./demo_data
```

### Output Files

**MegaHunter Output** (`output_smpl_and_points/`):
- `megahunter_{method}_reconstruction_results_{video}_cam01_frame_{start}_{end}_subsample_{factor}.h5`
  - `our_pred_world_cameras_and_structure`: Reconstructed world environment
  - `our_pred_humans_smplx_params`: SMPL parameters for each person
  - `person_frame_info_list`: Frame information for each person

**Final Results** (`output_calib_mesh/`):
- `gravity_calibrated_megahunter.h5`: Gravity-aligned human poses
- `gravity_calibrated_keypoints.h5`: 3D keypoints for all persons
- `background_mesh.obj`: Reconstructed environment mesh
- `background_less_filtered_colored_pointcloud.ply`: Less filtered point cloud
- `background_more_filtered_colored_pointcloud.ply`: Spatiotemporally filtered point cloud
- `retarget_poses_{robot_name}.h5`: Robot motion data


### Directory Structure

```
${PROJECT_ROOT}/
├── demo_data/               # Demo data for testing and visualization
│   ├── input_megasam/       # MegaSam reconstruction outputs
│   ├── input_align3r/       # Align3r reconstruction outputs
│   ├── input_images/
│   │   ├── people_jumping_nov20/
│   │   │   ├── cam01/
│   │   │   │   ├── 00001.jpg
│   │   │   │   ├── 00002.jpg
│   │   │   │   └── ...
│   ├── input_masks/         # SAM2 segmentation results
│   │   ├── people_jumping_nov20/
│   │   │   ├── cam01/
│   │   │   │   ├── mask_data/     # Binary masks
│   │   │   │   ├── json_data/     # Bounding boxes and metadata
│   │   │   │   └── meta_data.json # Multi-human tracking info
│   ├── input_2d_poses/      # ViTPose 2D pose results
│   │   ├── people_jumping_nov20/
│   │   │   ├── cam01/
│   │   │   │   ├── pose_00001.json  # 2D keypoints for each person
│   │   │   │   └── ...
│   ├── input_3d_meshes/     # VIMO/HMR2 3D mesh results
│   │   ├── people_jumping_nov20/
│   │   │   ├── cam01/
│   │   │   │   ├── smpl_params_00001.pkl  # SMPL parameters for each person
│   │   │   │   ├── known_betas.json        # Optimized shape parameters
│   │   │   │   └── ...
│   ├── input_contacts/      # BSTRO contact detection
│   │   ├── people_jumping_nov20/
│   │   │   ├── cam01/
│   │   │   │   ├── 00001.pkl
│   │   │   │   └── ...
│   ├── output_smpl_and_points/  # MegaHunter optimization results
│   └── output_calib_mesh/       # Final processed results
├── assets/
│   ├── checkpoints/
│   │   ├── align3r_depthpro.pth
│   │   ├── depth_pro.pt
│   │   ├── vitpose_huge_wholebody.pth
│   │   ├── vitpose_huge_wholebody_256x192.py
│   │   ├── hsi_hrnet_3dpw_b32_checkpoint_15.bin # bstro checkpoint for contact prediction
│   │   └── ...
│   ├── configs/
│   │   ├── bstro_hrnet_w64.yaml
│   │   ├── config_vimo.yaml
│   │   └── vitpose/
│   ├── body_models/      # SMPL/SMPLX models
│   ├── robot_asset/      # Robot URDF and assets
│   ├── ckpt_raft/        # RAFT optical flow checkpoints
│   └── ckpt_sam2/        # SAM2 model checkpoints
├── third_party/
│   ├── GeoCalib/         # Gravity calibration library
│   ├── Grounded-SAM-2/   # SAM2 segmentation
│   ├── ViTPose/          # 2D pose estimation
│   ├── megasam-package/  # MegaSam reconstruction
│   ├── monst3r-depth-package/ # Monst3r depth prior reconstruction
│   ├── NDC/              # Neural meshification (deprecated, replaced by NKSR)
│   ├── bstro/            # Contact detection
│   ├── VIMO/             # 3D human mesh estimation
├── stage0_preprocessing/
│   ├── sam2_segmentation.py
│   ├── vitpose_2d_poses.py
│   ├── vimo_3d_mesh.py
│   ├── bstro_contact_detection.py
│   ├── wilor_hand_poses.py
│   └── smpl_to_smplx_conversion.py
├── stage1_reconstruction/
│   ├── megasam_reconstruction.py
│   └── monst3r_depth_prior_reconstruction.py
├── stage2_optimization/
│   ├── optimize_smpl_shape_for_height.py
│   ├── optimize_smpl_shape_for_robot.py
│   ├── megahunter_optimization.py
│   ├── megahunter_costs.py        # JAX optimization costs
│   ├── megahunter_utils.py        # Utility functions
│   ├── megahunter_utils_robust.py # Robust multi-human utilities
│   └── README_robust_handling.md   # Multi-human handling docs
├── stage3_postprocessing/
│   ├── postprocessing_pipeline.py      # Main postprocessing script
│   ├── mesh_generation.py
│   ├── gravity_calibration.py
│   └── meshification.py
├── stage4_retargeting/
│   └── robot_motion_retargeting.py
├── sequential_processing/
│   ├── stage0_sequential_sam2_segmentation.py
│   ├── stage0_sequential_vitpose_2d_poses.py
│   ├── stage0_sequential_vimo_3d_mesh.py
│   ├── stage0_sequential_bstro_contact_detection.py
│   ├── stage1_sequential_megasam_reconstruction.py
│   ├── stage1_sequential_monst3r_depth_prior_reconstruction.py
│   ├── stage2_sequential_megahunter_optimization.py
│   └── stage3_sequential_mesh_generation_and_geocalib.py
├── visualization/
│   ├── sequential_visualization.py
│   ├── complete_results_egoview_visualization.py
│   ├── environment_only_visualization.py
│   ├── gravity_calibration_visualization.py
│   ├── mesh_generation_visualization.py
│   ├── optimization_results_visualization.py
│   ├── retargeting_visualization.py
│   ├── viser_camera_util.py
│   └── colors.txt           # Color palette for multi-human visualization
├── utilities/
│   ├── extract_frames_from_video.py
│   ├── smpl_jax_layer.py
│   ├── one_euro_filter.py
│   ├── egoview_rendering.py
│   └── viser_camera_utilities.py
├── docs/                 # Documentation
│   ├── setup.md
│   ├── commands.md
│   ├── directory.md
│   └── multihuman.md
├── video_scraping/       # Video collection and filtering tools
├── unit_tests/           # Unit tests for components
├── sloper4d_eval_script/ # Benchmark evaluation scripts
├── megahunter_models/    # Precomputed SMPL models
├── process_video.sh      # Main pipeline orchestration script
├── preprocess_human.sh   # Human preprocessing script
├── requirements.txt      # Python dependencies
└── README.md            # Project overview
```
