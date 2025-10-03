# Human Foot Position Fix

## Author
Qiushi Lyu (qlyu2@illinois.edu)

## Set up
- Conda is recommended to manage Python packages
- You need to set up HSfM environments. Check HSfM GitHub Repo for more info.
- You need to download HSfM checkpoints in `HSfM_RELEASE` directory(the `checkpoints` directory and `body_model` directory), please check HSfM GitHub Repo for more info. 

## How to run?
- There's some input examples in the `HSfM_RELEASE/demo_data` directory, you can also add more diverse inputs.

- (Optional) for new inputs, run:
```bash
cd HSfM_RELEASE
./run_hsfm.sh --img-dir $input_dir$ --out-dir $output_dir$ --person-ids "1 2 3" --vis
``` 
to get the HSfM outputs, the arg `person-ids` depends on the number of persons in the scene. There are already three examples in `HSfM_RELEASE/demo_data` directory, and their HSfM outputs are in `HSfM_RELEASE/demo_output` directory. If you don't want to add new inputs, just ignore this point.

- To run the implemented foot fix, please modify the **hsfm-pkl** and **out-pkl** args of `foot_fix/run.sh` to the HSfM output directory, and run `./foot_fix/run.sh` to get the foot fixed result. 

- You can visualize the result by `python vis_viser_hsfm.py --hsfm-pkl ./demo_output/people_jumping/hsfm_output_smplx.pkl` in the `HSfM_RELEASE` directory.
