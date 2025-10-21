# Changing the model architecture in GNS framework to Transformer

## Hacker: Sachidanand VS (sv69)

In This repo we try to implement a Transformer model with gloabl and local attention mechanism and see how well it can model the physics simulation processes. And check whether modelling physical simulation needs a model which has a lot of inductive bais of the system.

## Env setup
```shell
conda env create -f environment.yml
```

## Run GNS/transformer

> Training GNS/Transformer on simulation data
```shell
# For Orig GNS code,
python -m gns.train_transformer --data_path=./datasets/WaterDropSample/dataset/ --model_path=./models/GNS/waterDrop/ --output_path=./rollouts/GNS/waterDrop/ --model_type=gnn

# For transformer model,
python -m gns.train_transformer --data_path=./datasets/WaterDropSample/dataset/ --model_path=./models/transformer/waterDrop/ --output_path=./rollouts/transformer/waterDrop/ --model_type=transformer

# Change the dataset path as per the dataset requirement
```

> Rollout prediction
```shell
# For particulate domain,
python -m gns.train --mode="rollout" --data_path="<input-data-path>" --model_path="<path-to-load-save-model-file>" --output_path="<path-to-save-output>" --model_file="model.pt" --train_state_file="train_state.pt"

```

> Render
```shell
# For particulate domain,
python -m gns.render_rollout --output_mode="gif" --rollout_dir="<path-containing-rollout-file>" --rollout_name="<name-of-rollout-file>"
```

## Command line arguments details
<details>
<summary>`train.py` in GNS (particulate domain) </summary>

**mode (Enum)** 

This flag is used to set the operation mode for the script. It can take one of three values; 'train', 'valid', or 'rollout'.

**batch_size (Integer)**

Batch size for training.

**noise_std (Float)** 

Standard deviation of the noise when training.

**data_path (String)** 

Specifies the directory path where the dataset is located. 
The dataset is expected to be in a specific format (e.g., .npz files).
It should contain `metadata.json`.
If `--mode` is training, the directory should contain `train.npz`.
If `--mode` is testing (rollout), the directory should contain `test.npz`.
If `--mode` is valid, the directory should contain `valid.npz`.

**model_path (String)** 

The directory path where the trained model checkpoints are saved during training or loaded from during validation/rollout.

**output_path (String)** 

Defines the directory where the outputs (e.g., rollouts) are saved, 
when the `--mode` is set to rollout.
This is particularly relevant in the rollout mode where the predictions of the model are stored.

**output_filename (String)** 

Base filename to use when saving outputs during rollout.
Default is "rollout", and the output will be saved as `rollout.pkl` in `output_path`. 
It is not intended to include the file extension.

**model_file (String)** 

The filename of the model checkpoint to load for validation or rollout (e.g., model-10000.pt). 
It supports a special value "latest" to automatically select the newest checkpoint file. 
This flexibility facilitates the evaluation of models at different stages of training.

**train_state_file (String)** 

Similar to model_file, but for loading the training state (e.g., optimizer state).
It supports a special value "latest" to automatically select the newest checkpoint file. 
(e.g., training_state-10000.pt)

**ntraining_steps (Integer)** 

The total number of training steps to execute before stopping.

**nsave_steps (Integer)** 

Interval at which the model and training state are saved.

**lr_init (Float)** 

Initial learning rate.

**lr_decay (Float)** 

How much the learning rate should decay over time.

**lr_decay_steps (Integer)** 

Steps at which learning rate should decay.

**cuda_device_number (Integer)** 

Base CUDA device (zero indexed).
Default is None so default CUDA device will be used.

**n_gpus (Integer)** 

Number of GPUs to use for training.
</details>


## Datasets
### Particulate domain:
The Original codebase uses the numpy `.npz` format for storing positional data for GNS training.  The `.npz` format includes a list of tuples of arbitrary length where each tuple corresponds to a differenet training trajectory and is of the form `(position, particle_type)`.  The data loader provides `INPUT_SEQUENCE_LENGTH` positions, set equal to six by default, to provide the GNS with the last `INPUT_SEQUENCE_LENGTH` minus one positions as input to predict the position at the next time step.  The `position` is a 3-D tensor of shape `(n_time_steps, n_particles, n_dimensions)` and `particle_type` is a 1-D tensor of shape `(n_particles)`.  

The dataset contains:

* Metadata file with dataset information `(sequence length, dimensionality, box bounds, default connectivity radius, statistics for normalization, ...)`:

```
{
  "bounds": [[0.1, 0.9], [0.1, 0.9]], 
  "sequence_length": 320, 
  "default_connectivity_radius": 0.015, 
  "dim": 2, 
  "dt": 0.0025, 
  "vel_mean": [5.123277536458455e-06, -0.0009965205918140803], 
  "vel_std": [0.0021978993231675805, 0.0026653552458701774], 
  "acc_mean": [5.237611158734309e-07, 2.3633027988858656e-07], 
  "acc_std": [0.0002582944917306106, 0.00029554531667679154]
}
```
* npz containing data for all trajectories `(particle types, positions, global context, ...)`:

Training datasets for Sand, SandRamps, and WaterDropSample are available on [DesignSafe Data Depot](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published/PRJ-3702) [@vantassel2022gnsdata].

We provide the following datasets:
  * `WaterDropSample` (smallest dataset)
  * `Sand`
  * `SandRamps`

Download the dataset [DesignSafe DataDepot](https://doi.org/10.17603/ds2-0phb-dg64). If you are using this dataset please cite [Vantassel and Kumar., 2022](https://github.com/geoelements/gns#dataset)


### Acknowledgement
This code is build upon the [gns](https://github.com/geoelements/gns/tree/main) which has the original implementation of GNS framework. Thank you for their awesome work!
