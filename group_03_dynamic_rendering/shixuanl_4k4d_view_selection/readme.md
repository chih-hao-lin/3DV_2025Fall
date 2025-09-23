# 4K4D: Data-efficient Rendering with View Selection
### Hacker: Shixuan Liu (shixuanl@illinois.edu)

### Set up
Please follow the original 4K4D repository https://github.com/zju3dv/4K4D to set up the environments
A few key dependencies are
- `torch`
- `pytorch3d`
- `tinycudann`
- `open3d`

### How to run?
- Run `evc-tests -c configs/projects/realtime4dv/rendering/4k4d_0013_01.yaml,configs/specs/eval.yaml,configs/specs/spiral.yaml,configs/specs/ibr.yaml,configs/specs/vf0.yaml,configs/specs/video.yaml`