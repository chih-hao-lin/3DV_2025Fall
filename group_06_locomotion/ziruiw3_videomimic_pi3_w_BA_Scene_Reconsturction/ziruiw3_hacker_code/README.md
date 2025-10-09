# VideoMimic Real2Sim by Pi3 + BA for Scene Reconstruction

Hacker: Zirui Wang (ziruiw3@illinois.edu)

This is a modified version of VideoMimic Real2Sim pipeline, where we leverage better scene reconstruction model to acquire better aligned world information. Results is stored in `real2sim/demo_data/output_calib_mesh`, and can be visualized using Viser and the command line shown below. Example output can be found [here](https://drive.google.com/file/d/1IX6kFhAOSG5h4zJ_sbJmpcwqgjQ_guKN/view?usp=sharing).

## Set up
Follow the instruction in VideoMimic `real2sim` folder for detailed instructions. All additional pakages has been installed in the third-party folder. 

## 🚀 Quick Pipeline

```bash
# Prepare `demo_data` directory
# Extract frames from video (optional)
python utilities/extract_frames_from_video.py --video-path video.MOV --output-dir ./demo_data/input_images/video_name/cam01

# Run complete pipeline
cd real2sim
./process_video.sh <video_name> <start_frame> <end_frame> <subsample_factor> g1 <height>
# Example: ./process_video.sh my_video 0 100 2 g1 1.8
# height -1 uses auto-detected height from the video (default)
# height 0 uses the g1's shape

# Visualize results
conda activate vm1rs
python visualization/complete_results_egoview_visualization.py \
    --postprocessed-dir ./demo_data/output_calib_mesh/<result_dir> \
    --robot-name g1 --is-megasam
```


# VideoMimic

[[project page]](https://www.videomimic.net/) [[arxiv]](https://arxiv.org/pdf/2505.03729)  

**Visual Imitation Enables Contextual Humanoid Control. arXiV, 2025.**
    
<div style="background-color: #333; padding: 16px 20px; border-radius: 8px; color: #eee; font-family: sans-serif; line-height: 1.6;">
  <div style="font-size: 14px; margin-bottom: 12px;">
    Arthur Allshire<sup>*</sup>, Hongsuk Choi<sup>*</sup>, Junyi Zhang<sup>*</sup>, David McAllister<sup>*</sup>, 
    Anthony Zhang, Chung Min Kim, Trevor Darrell, Pieter Abbeel, Jitendra Malik, Angjoo Kanazawa (*Equal contribution) 
  </div>    
  <div style="font-size: 14px;">
    <i>University of California, Berkeley</i>
  </div>
</div>

## Updates

- **Sep 15, 2025:** Simulation code and preliminary sim2real code released.
- **Jul 6, 2025:** Initial real-to-sim pipeline release. 

## TODO

- [x] Release real‑to‑sim pipeline (July 15th, 2025)
- [x] Release the video dataset (July 15th, 2025) 
- [x] Release simulation pipeline (September 15th, 2025) 
- [x] Release sim2real code (September 15th, 2025) 

# VideoMimic Real-to-Sim

VideoMimic’s [real-to-sim pipeline](real2sim/README.md) reconstructs 3D environments and human motion from single-camera videos and retargets the motion to humanoid robots for imitation learning. It extracts human poses in world coordinates, maps them to robot configurations, and reconstructs environments as pointclouds later converted to meshes.

 
