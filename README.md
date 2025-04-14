# 🦿 LEG_GYM: Reinforcement Learning for Legged Robots with Isaac Gym

`LEG_GYM` is a reinforcement learning framework adapted from [legged_gym](https://github.com/leggedrobotics/legged_gym), designed for bipedal locomotion and extended with custom environments, curriculum strategies, and logging tools. It integrates NVIDIA Isaac Gym and RSL-RL.



## 📦 Features

- ✅ Isaac Gym + RSL-RL integration  
- 🦿 Biped locomotion in rough terrain  
- 📚 Modular curriculum and reward configuration  
- 🧠 Support for language-conditioned commands  
- 📈 WandB integration for experiment tracking  


## 🗂️ Project Structure

```
b1_gym/
├── legged_gym/                ← Main environment and training code
├── third_party/
│   └── rsl_rl/                ← Embedded RL framework (fork or submodule)
├── resources/                 ← URDFs, heightfields, config files
├── logs/                      ← Tensorboard or WandB logs
├── environment.yml           ← Conda environment file
├── setup.py                  ← Setup script (editable install)
├── LICENSE
└── README.md
```



## 🚀 Installation

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




### 4 Install Isaac Gym

Isaac Gym is **not included in this repository**. You must manually download and install it from NVIDIA:

#### 📥 1. Download Isaac Gym Preview 4

- Go to: [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
- You’ll need to create an NVIDIA Developer account to access the download.

> ⚠️ Isaac Gym is only supported on Linux + NVIDIA GPU + CUDA 11.x

#### 🧩 2. Install Isaac Gym Python bindings

```bash
cd isaacgym/python
pip install -e .
```

Make sure this directory is added to your `PYTHONPATH` or installed in editable mode.

#### 🧪 3. Test the installation (optional)

```bash
cd isaacgym/examples
python 1080_balls_of_solitude.py
```

This test should open a viewer with bouncing balls. If it crashes, check your GPU and driver setup.

#### ❓ 4. Troubleshooting

- Refer to: `isaacgym/docs/index.html`
- Or check the NVIDIA developer forums for platform-specific issues




## 🧪 Training Example

```bash
python legged_gym/scripts/train.py task=rough_b1
```

You can configure the task in `legged_gym/cfg/train/rough_b1.yaml`.



## 🎮 Evaluation

```bash
python legged_gym/scripts/play.py task=rough_b1
```




## 📝 Acknowledgements

- This project is based on [legged_gym](https://github.com/leggedrobotics/legged_gym)
- Reinforcement learning core from [rsl_rl](https://github.com/leggedrobotics/rsl_rl)



## 📜 License

This project is licensed under the **BSD 3-Clause License**. See [LICENSE](./LICENSE) for details.
