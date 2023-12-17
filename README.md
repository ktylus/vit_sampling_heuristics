# Adaptive Sampling Heuristics for Vision Transformers

Files description:

Folder **src** contains all source and utility python modules.

The script **train.py** is used for ViT model trainig with or without frozen params.
How to run it in the console, e.g.: `python train.py -lr 0.001 -e_n 50 -b_s 32`.

Default values are specified, hence running a command: `python train.py` is also valid. 

For more details and help run: `python train.py -h`.

For this project we recommend using Python 3.10 version. An appropriate environment can be installed accordingly:
`conda create --name <venv_name> python=3.10`

All the required packages for this project can be installed via the command:
`pip install -r requirements.txt` or using pip3 `pip3 install -r requirements.txt`

# Sources:
[1] [Beyond Grids: Exploring Elastic Input Sampling for Vision Transformers](https://arxiv.org/abs/2309.13353)<br/>
[2] [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)<br/>
[3] [Beyond Grids repository](https://github.com/apardyl/beyondgrids-next), Apache 2.0 license<br/>
[4] [Grad-CAM repository](https://github.com/jacobgil/pytorch-grad-cam), MIT license<br/>
[5] [Oxford Flowers102 Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)<br/>
[6] [Stanford Cars Dataset](https://pytorch.org/vision/main/generated/torchvision.datasets.StanfordCars.html)<br/>
