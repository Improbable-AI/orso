# ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization

This repository contains the code for the paper "ORSO: Accelerating Reward Design via Online Reward Selection and Policy Optimization".

<div align="center">

[arXiv](https://arxiv.org/abs/2410.13837) | [Website](https://calvincbzhang.github.io/orso-website/)

</div>

# Installation
ORSO requires Python ≥ 3.8

1. Create a new conda environment with
    ```
    conda create -n orso python=3.8
    conda activate orso
    ```

2. Install IsaacGym. Follow the [instruction](https://developer.nvidia.com/isaac-gym) to download the package.
    ```	
    tar -xvf IsaacGym_Preview_4_Package.tar.gz
    cd isaacgym/python
    pip install -e .
    (test installation) python examples/joint_monkey.py
    ```

3. Install ORSO
    ```
    pip install -e .
    cd isaacgymenvs
    pip install -e .
    ```

4. Set an environemnt variable for the OpenAI API key
    ```
    export OPENAI_API_KEY= "YOUR_API_KEY"
    ```

# Running Experiments

Navigate to the `src` directory and run:
```
python train_orso_budget.py env={env}
```
The full set of hyperparameters can be found in `src/config/config.yaml` and `src/config/envs/{env}.yaml` for environment specific parameters.

# Work in Progress
An implementation of ORSO with a fixed set of reward functions and without language model will be available soon. We will also provide a minimal implementation framework with CleanRL for practitioners to easily integrate ORSO into their projects.

# Citation
If you find this code useful, please consider citing our paper:
```bibtex
@inproceedings{
  zhang2025orso,
  title={{ORSO}: Accelerating Reward Design via Online Reward Selection and Policy Optimization},
  author={Chen Bo Calvin Zhang and Zhang-Wei Hong and Aldo Pacchiano and Pulkit Agrawal},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=0uRc3CfJIQ}
}
```
