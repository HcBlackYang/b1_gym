# b1_gym: Reinforcement Learning for Legged Robots with Isaac Gym

`b1_gym` is a reinforcement learning framework adapted from [legged_gym](https://github.com/leggedrobotics/legged_gym), designed for quadruped robot and extended with custom environments, curriculum strategies, and logging tools. It integrates NVIDIA Isaac Gym and rsl-rl.

## Environment Requirements

- **Operating System**: Ubuntu 20.04
- **Python Version**: 3.8
- **CUDA Version**: 11.8
- **PyTorch Version**: 2.4.1 + cu118
- **Other Dependencies**: See `environment.yml` 


## Features

-  Isaac Gym + rsl-rl integration  
-  Biped locomotion in rough terrain  
-  Modular curriculum and reward configuration  
-  Support for language-conditioned commands  



## Project Structure

```
b1_gym/
├── legged_gym/                ← Main environment and training code
├── third_party/
│   └── rsl_rl/                ← Embedded RL framework
├── resources/                 ← URDF
├── logs/                      ← Tensorboard
├── environment.yml           ← Conda environment file
├── setup.py                  ← Setup script
├── LICENSE
└── README.md
```



## Installation

### 1. Clone this repo

```bash
git clone https://github.com/HcBlackYang/b1_gym.git
cd b1_gym
```

### 2. Create Conda environment

```bash
conda env create -f environment.yml
conda activate b1_gym_env
```

### 3. Install local packages (editable mode)

```bash
pip install -e .
pip install -e ./third_party/rsl_rl
```

### 4. Install pytorch (with CUDA 11.8 support)

To install PyTorch 2.4.1 and compatible versions of torchvision and torchaudio with CUDA 11.8, run:

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```


### 5 Install Isaac Gym

Isaac Gym is **not included in this repository**. You must manually download and install it from NVIDIA:

#### 1. Download Isaac Gym Preview 4

- Go to: [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
- You’ll need to create an NVIDIA Developer account to access the download.

> Isaac Gym is only supported on Linux + NVIDIA GPU + CUDA 11.x

#### 2. Install Isaac Gym Python bindings

```bash
cd isaacgym/python
pip install -e .
```

Make sure this directory is added to your `PYTHONPATH` or installed in editable mode.

#### 3. Test the installation (optional)

```bash
cd isaacgym/examples
python 1080_balls_of_solitude.py
```

This test should open a viewer with bouncing balls. If it crashes, check your GPU and driver setup.







## Train:

```bash
python legged_gym/scripts/train.py --task=b1
```

To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).  
To run headless (no rendering) add `--headless`.  
**Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.  


The following command line arguments override the values set in the config files:

- `--task TASK`: Task name.  
- `--resume`: Resume training from a checkpoint  
- `--experiment_name EXPERIMENT_NAME`: Name of the experiment to run or load.  
- `--run_name RUN_NAME`: Name of the run.  
- `--load_run LOAD_RUN`: Name of the run to load when resume=True. If -1: will load the last run.  
- `--checkpoint CHECKPOINT`: Saved model checkpoint number. If -1: will load the last checkpoint.  
- `--num_envs NUM_ENVS`: Number of environments to create.  
- `--seed SEED`: Random seed.  
- `--max_iterations MAX_ITERATIONS`: Maximum number of training iterations.  

## Play a trained policy:

```bash
python legged_gym/scripts/play.py --task=b1
```

By default, the loaded policy is the last model of the last run of the experiment folder.  
Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.




##  Acknowledgements

- This project is based on [legged_gym](https://github.com/leggedrobotics/legged_gym)
- Reinforcement learning core from [rsl_rl](https://github.com/leggedrobotics/rsl_rl)



##  License

This project is licensed under the **BSD 3-Clause License**. See [LICENSE](./LICENSE) for details.
