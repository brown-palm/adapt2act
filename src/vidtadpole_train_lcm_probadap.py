import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import StridedOverlappingEpisode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'

from utils import set_seed, AnimateLCMProbAdaptation


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards, episode_success = [], []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video: video.init(env, enabled=(i==0))
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        episode_success.append(int(info.get('success', 0)))
        if video: video.save(env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_success)


def train(cfg):
    """Adapt2Act training script for Video-TADPoLe TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
    device = torch.device('cuda')

    domain, task = cfg.task.replace('-', '_').split('_', 1)
    camera_id = dict(quadruped=2).get(domain, 0)
    dim = dict(dog=512, metaworld=512).get(domain, 480)
    render_kwargs = dict(height=dim, width=dim, camera_id=camera_id)

    noise_level = cfg.noise_level
    align_scale = cfg.alignment_scale
    recon_scale = cfg.recon_scale
    context_size = cfg.context_window
    stride = cfg.stride

    guidance = AnimateLCMProbAdaptation(device, domain=cfg.domain)

    negative_prompts = "bad quality, worse quality"
    c_in = guidance.get_text_embeds(cfg.text_prompt, negative_prompts)

    if cfg.text_prompt is not None and isinstance(cfg.text_prompt, str):
        batch_size = 1
    elif cfg.text_prompt is not None and isinstance(cfg.text_prompt, list):
        batch_size = len(cfg.text_prompt)

    avdc_prompt = cfg.avdc_prompt if isinstance(cfg.avdc_prompt, list) else [cfg.avdc_prompt] * batch_size
    avdc_text_embeddings = guidance.avdc_trainer.encode_batch_text(avdc_prompt)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()

    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        obs = env.reset()
        episode = StridedOverlappingEpisode(cfg, obs, context_size, stride=stride)

        rendered = torch.Tensor(env.render(**render_kwargs).copy()[np.newaxis, ...]).permute(0,3,1,2).to(device)
        latent = guidance.encode_imgs(rendered.half() / 255.)
        latent_arr = [latent]
        latent_history = latent.repeat(context_size, 1, 1, 1)

        success_history = torch.zeros([context_size], dtype=torch.float32).to(device)
        timestep = torch.randint(noise_level, noise_level + 100, [1], dtype=torch.long, device=device)
        while not episode.done:
            source_noise = torch.randn_like(latent_history) # source noise to be the same as latent history, resampled per ts
            action = agent.plan(obs, step=step, t0=episode.first)
            obs, gt_reward, done, info = env.step(action.cpu().numpy())
            rendered = torch.Tensor(env.render(**render_kwargs).copy()[np.newaxis, ...]).permute(0,3,1,2).half().to(device)
            latent = guidance.encode_imgs(rendered / 255.)

            latent_history = latent_history.roll(-1, 0)
            latent_history[-1] = latent

            success_history = success_history.roll(-1, 0)
            success_history[-1] = float(info.get('success', 0))

            reward = torch.zeros([context_size], dtype=torch.float32).to(device)
            if ((episode._idx + 2 - context_size) % stride == 0 or episode._idx == cfg.episode_length - 1) and episode._idx >= (context_size-2):
                reward += guidance.get_videotadpole_reward(c_in, latent_history, timestep, alignment_scale=align_scale, recon_scale=recon_scale, noise=source_noise, prior_strength=cfg.prior_strength, avdc_text_embeddings=avdc_text_embeddings, avdc_cond=latent_arr[0] if episode._idx + 1 <= context_size else latent_arr[episode._idx + 1 - context_size], inverted_probadap=cfg.inverted_probadap)
                reward += cfg.sparse_scale * success_history

            episode += (obs, action, reward, gt_reward, done)
            latent_arr.append(latent)
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step+i))

        # Log training episode
        episode_idx += 1
        env_step = int(step*cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'episode_gt_reward': episode.cumulative_gt_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train', agent=agent)

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            eval_episode_gt_reward, eval_success_rate = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            eval_metrics = {
                'env_step': env_step,
                'episode_gt_reward': eval_episode_gt_reward,
                'success_rate': eval_success_rate
            }
            L.log(eval_metrics, category='eval', agent=agent)

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
