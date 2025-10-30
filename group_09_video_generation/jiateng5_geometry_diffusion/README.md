# Topic: Geometry Forcing Diffusion Models

### Project: jiateng5_geometry_diffusion

### Instruction for setting up the environment:
```
conda create -n geometryforcing python=3.10 -y
conda activate geometryforcing
pip install -r requirements.txt
```

### Download Checkpoints and Data
1. Download pretrained checkpoint using Hugging Face:
```
bash scripts/hf_download_checkpoints.sh
```

2. Download pretrained checkpoint using ModelScope:
```
bash scripts/ms_download_checkpoints.sh

3. Download and process RealEstate10K dataset to `data/real-estate-10k` using the helper script:
```
python data/download_video.py
```

If your RealEstate10K metadata needs fixing (paths/layout), use:
```
python fix_metadata.py
```

### Running the scripts
- Run evaluation (Single Image to Long Video):
```
bash scripts/eval_geometry_forcing.sh
```

- Run training:
```
bash scripts/train_geometry_forcing.sh
```

### Example outputs
Generated examples are available under:
```
examples/prediction_vis
```