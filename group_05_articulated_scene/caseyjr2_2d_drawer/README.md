# 2D DRAWER Implementation - Hacker Casey Rodgers (caseyjr2)
## Using the code
1.) Download the code.<br/>
2.) Set up the environment as detailed below. <br/>
3.) Download the 3DOI files here](https://github.com/JasonQSY/3DOI/tree/main?tab=readme-ov-file) and copy them into a folder called "threedoi" in the same folder as the main .ipynb file. <br/>
4.) Download the 3DOI pretrained checkpoint, which can be found [here](https://github.com/JasonQSY/3DOI/tree/main?tab=readme-ov-file). The file name should be "checkpoint_20230515.pth". Put this file in the same folder as the main .ipynb file.<br/>
5.) Copy and paste the "merged.yaml" file into the "configs" folder in the 3DOI folder (threedoi -> monoarti -> configs). <br/>
6.) Open the .ipynb file with Jupyter notebook and run all of the cells. Make sure to have your new environment enabled as the kernel.<br/>
7.) With all of the packages installed, it should run smoothly. Note, it will take a few minutes to run all of the cells.<br/>

## Set up the environment
Create a python enviornment with all of the needed packages
```
# python
conda create -n monoarti python=3.9
conda activate monoarti

# pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# other packages
pip install accelerate
pip install submitit
pip install hydra-core --upgrade --pre
pip install hydra-submitit-launcher --upgrade
pip install pycocotools
pip install packaging plotly imageio imageio-ffmpeg matplotlib h5py opencv-python
pip install tqdm wandb visdom

# More packages
pip install transformers
```

If you get an error about a missing package when trying to import, then install those as well.