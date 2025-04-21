import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
gym.logger.set_level(40)
from pathlib import Path
from cfg import parse_cfg
from env import make_env
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

from utils import set_seed, probadap_sampling, InvDynamics

from omegaconf import OmegaConf
from termcolor import colored


def rollout(cfg):
    """Adapt2Act visual planning rollout script. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    os.makedirs(work_dir, exist_ok=True)
    env = make_env(cfg)
    device = torch.device('cuda')

    domain, task = cfg.task.replace('-', '_').split('_', 1)
    camera_id = dict(quadruped=2).get(domain, 0)
    dim = dict(dog=512, metaworld=512).get(domain, 480)
    render_kwargs = dict(height=dim, width=dim, camera_id=camera_id)

    animatediff_pipe = None
    avdc_trainer = None
    if cfg.plan_with_probadap:
        from utils import AnimateDiffProbAdaptation
        guidance = AnimateDiffProbAdaptation(device, domain=cfg.domain, use_suboptimal=cfg.use_suboptimal)
        animatediff_pipe = guidance.pipe
        avdc_trainer = guidance.avdc_trainer
    elif cfg.plan_with_dreambooth:
        from utils import AnimateDiffDreamBooth
        guidance = AnimateDiffDreamBooth(device, domain=cfg.domain)
        animatediff_pipe = guidance.pipe
    elif cfg.plan_with_finetuned:
        from utils import AnimateDiffDirectFT
        guidance = AnimateDiffDirectFT(device, domain=cfg.domain)
        animatediff_pipe = guidance.pipe
    else:
        from utils import AnimateDiff
        guidance = AnimateDiff(device)
        animatediff_pipe = guidance.pipe

    #instantiate inv_model
    inv_model = InvDynamics()
    inv_model = inv_model.to(device)

    # load inv_model checkpoint weights
    print('[INFO] loading inverse dynamics model checkpoint...')
    inv_checkpoint = torch.load(cfg.inv_ckpt_path, map_location="cpu")
    m, u  = inv_model.load_state_dict(inv_checkpoint['inv_model'], strict=False)
    print('[INFO] loaded inverse dynamics model checkpoint!')

    # get prompts, avdc prompt, etc.
    animatediff_prompt = cfg.text_prompt
    negative_prompt = "bad quality, worse quality"
    if avdc_trainer:
        cfg.avdc_prompt = ' '.join(cfg.task.split('-')[1:])

    project, entity = cfg.get('wandb_project', 'none'), cfg.get('wandb_entity', 'none')
    run_offline = not cfg.get('use_wandb', False) or project == 'none' or entity == 'none'
    run_name = f"{cfg.task}__{cfg.exp_name}__{cfg.text_prompt}__seed_{cfg.seed}__{int(time.time())}"

    wandb_instance = None
    if not run_offline:
        try:
            os.environ["WANDB_SILENT"] = "true"
            import wandb
            print(f'working dir:{work_dir}')
            wandb.init(project=project,
                    entity=entity,
                    name=run_name,
                    dir=work_dir,
                    config=OmegaConf.to_container(cfg, resolve=True))
            print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
            print(f'wandb link: {wandb.run.get_url()}')
            wandb_instance = wandb
        except:
            print(colored('Warning: failed to init wandb. Logs will be saved locally.', 'yellow'), attrs=['bold'])
            wandb_instance = None

    num_frames = 9 if cfg.plan_is_conditioning_animatediff else 8
    episode_success_flags = []
    episode_returns = []

    for rollout_idx in range(cfg.plan_num_rollouts):
        # begin loop
        obs, done, ep_reward = env.reset(), False, 0
        rendered = torch.Tensor(env.render(**render_kwargs).copy()[np.newaxis, ...] / 255.0).permute(0,3,1,2).to(device)
        orig_frame = F.interpolate(rendered, (512, 512), mode='bilinear', align_corners=False)

        # to store gt env renders and imagined frame future the policy follows
        env_frames = [orig_frame]
        imagined_frames = [orig_frame]
        actions = []

        while not done:
            if avdc_trainer:
                video = probadap_sampling(
                    animatediff_pipe=animatediff_pipe,
                    avdc_trainer=avdc_trainer,
                    animatediff_prompt=animatediff_prompt,
                    avdc_prompt=cfg.avdc_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    is_conditioning_animatediff=cfg.plan_is_conditioning_animatediff,
                    conditioning_frames=rendered,
                    num_inference_steps=cfg.plan_inf_steps,
                    guidance_scale=cfg.guidance_scale,
                    prior_strength=cfg.prior_strength,
                    output_type='pt',
                    is_inverted=cfg.inverted_probadap
                ).frames[0]
            else:
                video = animatediff_pipe(
                    prompt=animatediff_prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=cfg.plan_inf_steps,
                    conditioning_frames=rendered,
                    guidance_scale=cfg.guidance_scale,
                    output_type='pt',
                ).frames[0]
            
            video = video[1:, ...]  if cfg.plan_is_conditioning_animatediff else video # get rid of the first conditioning frame, range [0, 1]

            full_frames = torch.concat([orig_frame, video])
            assert cfg.plan_count < full_frames.shape[0]

            for i in range(cfg.plan_count):
                imagined_frames.append(full_frames[i+1:i+2])

                inv_input = full_frames[i:i+2]
                with torch.no_grad():
                    action = inv_model(inv_input[None,:].to(device))[0]

                obs, reward, done, info = env.step(action.cpu().numpy())
                ep_reward += reward
                rendered = torch.Tensor(env.render(**render_kwargs).copy()[np.newaxis, ...] / 255.0).permute(0,3,1,2).to(device)
                orig_frame = F.interpolate(rendered, (512, 512), mode='bilinear', align_corners=False)
                env_frames.append(orig_frame)
                actions.append(action.cpu().numpy())
                done = done or int(info.get('success', 0))
                
                if done:
                    episode_success_flags.append(float(info.get('success', 0)))
                    episode_returns.append(ep_reward)
                    break

        if wandb_instance:
            achieved_video = (torch.concatenate(env_frames) * 255.0).to(torch.uint8).cpu()
            wandb_instance.log({'eval_video': wandb_instance.Video(achieved_video, fps=15, format='mp4')}, step=rollout_idx)
            imagined_video_plan = (torch.concatenate(imagined_frames) * 255.0).to(torch.uint8).cpu()
            wandb_instance.log({'imagined_video_plan': wandb_instance.Video(imagined_video_plan, fps=15, format='mp4')}, step=rollout_idx)

            wandb_instance.log({'episode_reward': ep_reward, 'episode_success': int(info.get('success', 0))}, step=rollout_idx)
        print(f'Rollout {rollout_idx} completed successfully')


    if wandb_instance:
        wandb_instance.log({'avg_episode_reward': np.mean(episode_returns), 'avg_success_rate': np.mean(episode_success_flags)})



if __name__ == '__main__':
    rollout(parse_cfg(Path().cwd() / __CONFIG__))
