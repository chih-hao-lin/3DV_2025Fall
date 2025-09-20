# Infer the Unseen

Hacker: Bingxuan Li (bl61@illinois.edu)

## Project Structure

- `text_constrainted_inference.py` - Main inference script
- `environment.yml` - Conda environment configuration
- `run.sh` - Convenience script to run inference
- `src/` - Source code and model files
- `example/` - Example input images


## Set up

### Download Checkpoint

```bash
cd src
# for 224 linear ckpt
gdown --fuzzy https://drive.google.com/file/d/11dAgFkWHpaOHsR6iuitlB_v4NFFBrWjy/view?usp=drive_link 
# for 512 dpt ckpt
gdown --fuzzy https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view?usp=drive_link
```

### Environment Setup
```bash
conda env create -f environment.yml
conda activate cut3r
```

### OpenAI API Key Setup (optional)
Create a file named `.env` in the project root directory and add your OpenAI API key

## How to run?

### Choose the Generation Model

To switch between the "gpt" and "flux" generation models, edit the following line in `text_constrainted_inference.py`:

### Run

#### Option 1: Use the run script
```bash
./run.sh
```

#### Option 2: Run manually
```bash
python3 text_constrainted_inference.py <SESSION_ID> <WORKING_DIR> <MODEL_PATH>
```

Example:
```bash
python3 text_constrainted_inference.py 1 ./output ./src/cut3r_512_dpt_4_64.pth
```