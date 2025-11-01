# LoRA finetuning Wan2.2 with descriptive relative pose prompts 

Tested on A100 on colab pro with 80 gb VRAM

## Rosbag to dataset generation

### Create a venv 
```
python -m venv venv
source venv/bin/activate 

pip install -r requirements.txt

```

### OPTIONAL: Rosbag to PoV MP4 & odometry CSV
The dataset converted to mp4 and odometry csv with timestamps is uploaded on huggingface [link](https://huggingface.co/datasets/adipotnis/uiuc_south_quad), to convert additional data, try the WayFAST dataset [Link](https://github.com/matval/WayFAST?tab=readme-ov-file#download-our-dataset).  

For manually converting use:
```
cd dataset_prep
python extract_rgb_odom.py <rosbag folder path>/<rosbag name.bag>
```

The odom csv will be of form `<rosbag_name>_odom.csv` and the MP4 will be of form `<rosbag_name>_zed_left.mp4`. Place them in a folder with name `bags`.

### OPTIONAL: MP4 & CSV to text & dataset clips

The `plot_odom.py` file can be used to visualize the overall path that the robot has taken. it plots all the <rosbag>_odom.csv files in the `bags` folder. 

Convert the mp4 & odom csv to a huggingface ready dataset using `mp4_csv_to_dataset.py` 
Keep the mp4 and odom csv in `bags` folder. 
Generate api token for free on gemini ai studio [link](https://aistudio.google.com/api-keys)
```
export GEMINI_API_KEY=<gemini api token>
python mp4_csv_to_dataset.py
python create_metadata.py
mv clips ../south_quad_dataset
```
This should create a dataset with appropriate metadata for fine tuning the model. 


## LoRA finetuning Wan 2.2 

cd into the `lora_training` directory. 

deactivate the previous python venv, this training uses the [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) framework for the LoRA pipeline.  

1. Create env
```bash
conda create -n diffsynth python=3.12
```
2. Activate env
```bash
conda activate diffsynth
```
3. Clone
```bash
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
cd ..
```
4. Setup accelerate config
```bash
cp accelerate_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
# to verify
accelerate env
accelerate config show
```
5. Run training
Run the training script using `./lora_ft_wan_1_3B.sh`


6. Run inference
run the script `generate_video_yaml.py`
```
python generate_video_yaml.py
```