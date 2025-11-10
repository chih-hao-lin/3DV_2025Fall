## Set up environment
- Set up environment for physgen https://github.com/stevenlsw/physgen and GO-with-the-Flow https://github.com/Eyeline-Labs/Go-with-the-Flow 

## Prepare data
- Prepare an image in physgen/data/<duck_ball>. Change <duck_ball> to your own project name
- Generate intermediate files (sin.yaml, inpaint, mask, and obj_movable) using SAM referring to physgen github.
## Run physgen

    ```Shell
    export PYTHONPATH=$(pwd)
    name="duck_ball"
    cd physgen
    python simulation/animate.py --data_root data --save_root outputs --config data/${name}/sim.yaml
    ```
-  Copy config_mask.yaml, mask.png, original.png to Go-with-the-Flow/duck_ball

## Run Go-with-the-Flow
- Generate video with physics-based movements.
    ```Shell
    python cut_and_drag_gui.py
    ```

- Wrap noise
    ```Shell
    python make_warped_noise.py ${name}/${name}.mp4 --output_folder ${name}/${name}_warped_noise
    ```
- Run video diffusion
    ```Shell
    python cut_and_drag_inference.py ${name}/${name}_warped_noise\
        --prompt "A toy pig hit a toy ball" \
        --output_mp4_path "${name}/final_output.mp4" \
        --device "cuda" \
        --num_inference_steps 30
    ```
