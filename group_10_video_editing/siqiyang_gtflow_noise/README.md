# Go-with-the-Flow: Segmentation-Based Motion Control

## Hacker: Siqi Yang (siqiyang@illinois.edu)

## Setup

```bash
pip install -r requirements.txt
pip install -r requirements_local.txt
```

## How to run?

```
python cut_and_drag_gui.py
```

Follow the instructions shown in the GUI

You can use the provided duck_original folder to reproduce my experiment setup.

I reimplemented noise_warp.py for improved control and reproducibility.
You can follow the notes in noise_warp.py to run custom experiments.

After generating the warped noise, run inference with:

```
python cut_and_drag_inference.py noise_warp_output_folder \
    --prompt "A yellow rubber duck floating on calm blue water in a pool. The duck moves smoothly from left to right, creating gentle ripples around it. After reaching the side, it pauses briefly, then moves back from right to left. The water remains calm and slightly reflective. The background stays still." \
    --output_mp4_path "output.mp4" \
    --device "cuda" \
    --num_inference_steps 20
```

This command generates an animated video based on the specified motion prompt.

## Acknowledgements

- Original Go-with-the-Flow: [Project Page](https://eyeline-labs.github.io/Go-with-the-Flow/) | [Paper](https://arxiv.org/abs/2501.08331)

