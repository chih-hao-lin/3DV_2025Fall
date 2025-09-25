CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python cosmos_predict1/diffusion/inference/inference_inverse_renderer.py \
    --checkpoint_dir checkpoints --diffusion_transformer_dir Diffusion_Renderer_Inverse_Cosmos_7B \
    --dataset_path=asset/4k4d_inputs_2/dance3 --num_video_frames 57 --group_mode folder \
    --video_save_folder=asset/4k4d_outputs_2/dance3 --offload_diffusion_transformer --offload_tokenizer 