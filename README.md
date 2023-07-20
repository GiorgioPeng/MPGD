# `M`ixed `P`robabilistic Generation Model induced `G`raph `D`isentanglement (MPGD)

# Installation

## Pull
```
git clone https://github.com/GiorgioPeng/MPGD.git
```

## Create Virtual Environment (With **anaconda**)
```
conda create -n env_name python=3.6
```

## Install Dependencies 
```
pip install numpy==1.19.5

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

pip install scikit-learn==0.24.2

pip install networkx==2.5.1

pip install hyperopt==0.2.7
```

# Getting Started
- Activate the virtual environment (With **anaconda**)
```
conda activate env_name
```
- Run the command in `./example_sh/example.sh`

# Data
The `./data/` folder includes both standard benchmarks (Cora, Citeseer, Pubmed, Amazon Photo, Chameleon, Squirrel and Crocodile).