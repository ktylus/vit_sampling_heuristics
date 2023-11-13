# vit_sampling_heuristics

Files description:

Folder **src** contains all source and utility python modules.

The script **train.py** is used for ViT model trainig with or without frozen params.
How to run it in the console, e.g.: `python train.py -lr 0.001 -e_n 50 -b_s 32`.

Default values are specified, hence running a command: `python train.py` is also valid. 

For more details and help run: `python train.py -h`.

For this project we recommend using Python 3.9 version. An appropriate environment can be installed accordingly:
`conda create --name <venv_name> python=3.9`

All the required packages for this project can be installed via the command:
`pip install -r requirements.txt` or using pip3 `pip3 install -r requirements.txt`
