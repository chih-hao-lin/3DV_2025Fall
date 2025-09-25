# Relightable 4K4D

## Hacker: Jingwang Ling (ling23@illinois.edu)

## Set up

- Follow the [Original 4K4D README](./readme_old.md) to setup the Python environment. `requiresments-devel.txt` is updated to reflect the additional package requirement.
- Download the [Minimal Dataset](https://drive.google.com/drive/folders/1pH-SWwbt01raqZ74dvcOvYFxDbGGUcxu) and the [pretrained model](https://drive.google.com/drive/folders/1mBMsYeXawU_sF3NFyuWC1hnfrYbSfDfi) of `dance3`.
- Process images and masks using
```bash
python scripts/realtime4dv/extract_images.py --data_root data/mobile_stage/dance3
python scripts/realtime4dv/extract_images.py --data_root data/mobile_stage/dance3  --videos_dir videos_masks_libx265 --images_dir masks_libx265 --single_channel
```
- Preprocess DiffusionRenderer inputs `python scripts/diffusionrenderer/prepare_input.py --data_root data/mobile_stage/dance3 --output_root ../cosmos1-diffusion-renderer/asset/4k4d_inputs/dance3`.
- Set up DiffusionRenderer, go to DiffusionRenderer and run `bash ../4K4D/scripts/diffusionrenderer/run_diffrenderer.sh`.
- Gather DiffusionRenderer output `python scripts/diffusionrenderer/prepare_output.py --data_root data/mobile_stage/dance3 --output_root ../cosmos1-diffusion-renderer/asset/4k4d_outputs/dance3`.

## How to run?

- Run `bash scripts/diffusionrenderer/test_video_novel_light.sh` to rendering videos in novel lighting.
- Run `bash scripts/diffusionrenderer/test_video_original_light.sh` to rendering videos in original lighting.
- Video results can be shown in the [online slides](https://uillinoisedu-my.sharepoint.com/:p:/g/personal/ling23_illinois_edu/EW5hPfLOKtZItKMGp4NFjloB63yGbQvfyPqKYBJCzTjaoA?e=UYKFGe).

## Acknowledgement

This project is based on [4K4D](https://github.com/zju3dv/4K4D) and [nvdiffrec](https://github.com/NVlabs/nvdiffrec). Thanks for their impressive works.
