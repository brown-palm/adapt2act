# Solving New Tasks by Adapting Internet Video Knowledge
## Quick Start
### Setup Conda Environment
```sh
conda create -n adapt2act python=3.9
conda activate adapt2act

pip install pip==21.0 wheel==0.38.0 setuptools==65.5.0  # specified gym version requires these tools to be old
pip install -r requirements.txt
```

### Install Visual-Cortex
```sh
git clone https://github.com/facebookresearch/eai-vc.git
cd eai-vc

pip install -e ./vc_models
```

### Customize Configurations for DeepMind Control Environments
Please follow the same instruction in [TADPoLe](https://github.com/brown-palm/tadpole?tab=readme-ov-file#customize-configurations-for-dog-and-humanoid-environments) to customize configurations for Dog and Humanoid environments.

### Checkpoints
Please put `checkpoints/` under the `adapt2act/` folder, and the directory should have the following structure:
```
adapt2act/
└── checkpoints/
    ├── animatediff_finetuned/
    │   ├── {domain}_finetuned.ckpt
    │   └── ...
    ├── in_domain/
    │   ├── {domain}/
    │   ├── {domain}_suboptimal/
    │   └── ...
    ├── dreambooth/
    │   ├── {domain}_lora/
    │   └── ...
    └── inv_dyn.ckpt
```
We currently support three domains: Metaworld `mw`, Humanoid `humanoid` and Dog `dog`. The model checkpoints can be downloaded [here](https://drive.google.com/file/d/1bDoQq_z605cX6czWGKVRLnSgXjR050A8/view?usp=drive_link).

## Policy Supervision
> [!TIP]
> To enable wandb logging, enter your wandb entity in `cfgs/default.yaml` and add `use_wandb=True` to the commands below
### AnimateDiff
Vanilla AnimateDiff
```shell
python src/vidtadpole_train.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=False \
    use_finetuned=False
```

Direct Finetuning
```shell
python src/vidtadpole_train.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=False \
    use_finetuned=True
```

Subject Customization
```shell
python src/vidtadpole_train.py task="metaworld-door-close" \
    text_prompt="a [D] robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=True \
    use_finetuned=False
```

Probabilistic Adaptation
```shell
python src/vidtadpole_train_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.1 \
    inverted_probadap=False
```


Inverse Probabilistic Adaptation
```shell
python src/vidtadpole_train_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.1 \
    inverted_probadap=True
```

### AnimateLCM
Vanilla AnimateLCM
```shell
python src/vidtadpole_train_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    use_dreambooth=False \
    use_finetuned=False
```

Probabilistic Adaptation
```shell
python src/vidtadpole_train_lcm_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.1 \
    inverted_probadap=False
```

Inverse Probabilistic Adaptation
```shell
python src/vidtadpole_train_lcm_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0.2 \
    inverted_probadap=True
```

### In-Domain-Only
```shell
python src/vidtadpole_train_probadap.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    avdc_prompt="door close" \
    seed=0 \
    prior_strength=0 \
    inverted_probadap=False
```

## Visual Planning

### AnimateDiff

Vanilla AnimateDiff
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=7.5 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=False
```

Direct Finetuning
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=8 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=True
```

Subject Customization
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a [D] robot arm closing a door" \
    seed=0 \
    guidance_scale=7.5 \
    plan_with_probadap=False \
    plan_with_dreambooth=True \
    plan_with_finetuned=False
```


Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.1 \
    inverted_probadap=False
```

Inverse Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.5 \
    inverted_probadap=True
```

### AnimateLCM

Vanilla AnimateLCM
```shell
python src/visual_planning_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=False \
    plan_with_dreambooth=False \
    plan_with_finetuned=False
```


Probabilistic Adaptation
```shell
python src/visual_planning_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.1 \
    inverted_probadap=False
```

Inverse Probabilistic Adaptation
```shell
python src/visual_planning_lcm.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.2 \
    inverted_probadap=True
```

### In-Domain-Only
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0 \
    inverted_probadap=False
```

## Visual Planning with Suboptimal Data

In-Domain-Only
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0 \
    inverted_probadap=False \
    use_suboptimal=True
```

Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.1 \
    inverted_probadap=False \
    use_suboptimal=True
```

Inverse Probabilistic Adaptation
```shell
python src/visual_planning.py task="metaworld-door-close" \
    text_prompt="a robot arm closing a door" \
    seed=0 \
    guidance_scale=2.5 \
    plan_with_probadap=True \
    plan_with_dreambooth=False \
    plan_with_finetuned=False \
    prior_strength=0.5 \
    inverted_probadap=True \
    use_suboptimal=True
```


## Citation
If you find this repository helpful, please consider citing our work:
```bibtex
@inproceedings{luo2024solving,
  title={Solving New Tasks by Adapting Internet Video Knowledge},
  author={Luo, Calvin and Zeng, Zilai and Du, Yilun and Sun, Chen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Acknowledgement
This repo contains code adapted from [flowdiffusion](https://github.com/flow-diffusion/AVDC), [TDMPC](https://github.com/nicklashansen/tdmpc) and [TADPoLe](https://github.com/brown-palm/tadpole). We thank the authors and contributors for open-sourcing their code.
