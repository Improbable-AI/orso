# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy

import datetime
import logging
import os
import random
import sys
import time
from collections import deque

import gym
import hydra
import isaacgym  # noqa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import isaacgymenvs
from agent import Agent
from utils.schedulers import AdaptiveScheduler
from utils.agent_utils import RunningMeanStd
from utils.misc import *
from utils.wrappers import ExtractObsWrapper, RecordEpisodeStatisticsTorch

SRC_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{SRC_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

sns.set_theme(
    context="paper",
    style="whitegrid",
    rc={
        "lines.linewidth": 3,
    },
    font_scale=1.5,
)

np.set_printoptions(linewidth=np.inf)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    target_task = cfg.env.task  # ShadowHand
    target_name = cfg.env.name  # shadow_hand

    network_params = cfg.env.network
    train_params = cfg.env.train

    keys_to_ignore = ["consecutive_successes", "time_outs", "r", "l"]

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"{target_task}_{timestamp}"
    log_dir = cfg.log_dir

    print(f"Log directory: {log_dir}")

    # Print configuration
    print("======== Configuration ========")
    print(OmegaConf.to_yaml(cfg))
    print("=" * len("======== Configuration ========"))

    budget = cfg.remaining_budget - 1
    print(f"REMAINING BUDGET: {budget}")

    gpus = set_freest_gpu()

    try:
        wandb_tags = cfg.wandb_tags
    except:
        wandb_tags = ["eureka"]
    wandb_group = cfg.wandb_group

    batch_size = int(train_params.num_envs * train_params.num_steps)
    minibatch_size = int(batch_size // train_params.num_minibatches)
    num_iterations = train_params.total_timesteps // batch_size

    print(f"Batch size: {batch_size}")
    print(f"Minibatch size: {minibatch_size}")
    print(f"Number of iterations: {num_iterations}")

    if cfg.use_wandb:
        import wandb

        wandb_unique_id = f"uid_{run_name}"
        print(f"WandB using unique id {wandb_unique_id}")

        wandb.init(
            dir=log_dir,
            project=cfg.wandb_project,
            entity=cfg.wandb_username,
            group=wandb_group,
            tags=wandb_tags,
            sync_tensorboard=True,
            monitor_gym=True,
            id=wandb_unique_id,
            name=run_name,
            config=vars(train_params),
            resume=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{log_dir}/runs/{run_name}/summaries")
    print(f"Tensorboard Directory: {log_dir}/runs/{run_name}/summaries")

    os.makedirs(f"{log_dir}/runs/{run_name}/nn", exist_ok=True)

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = isaacgymenvs.make(
        seed=cfg.seed,
        task=f'{target_task}{cfg.suffix}',
        num_envs=train_params.num_envs,
        sim_device=f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu",
        rl_device=f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu",
        graphics_device_id=int(gpus[0]) if torch.cuda.is_available() else -1,
        headless=True,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,
    )
    if cfg.capture_video:
        envs.is_vector_env = True
        envs = gym.wrappers.RecordVideo(
            envs,
            f"{log_dir}/videos/{run_name}",
            step_trigger=lambda step: step % cfg.capture_video_freq == 0,
            video_length=cfg.capture_video_len,
        )
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    reward_normalizer = RunningMeanStd((1,)).to(device)

    agent = Agent(envs, network_params, train_params.norm_input).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=train_params.learning_rate, eps=1e-5)
    if train_params.lr_schedule == "adaptive":
        scheduler = AdaptiveScheduler(kl_threshold=train_params.kl_threshold)
    scaler = torch.cuda.amp.GradScaler(enabled=train_params.mixed_precision)

    # ALGO LOGIC: storage setup
    obs = torch.zeros((train_params.num_steps, train_params.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((train_params.num_steps, train_params.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_done = torch.zeros(train_params.num_envs, dtype=torch.float).to(device)

    best_cs = -np.inf

    done_mean_ep_returns = deque(maxlen=train_params.num_steps)
    done_mean_ep_len = deque(maxlen=train_params.num_steps)
    done_ep_cses = deque(maxlen=train_params.num_steps)

    for iteration in range(1, num_iterations + 1):
        print(f"======== REMAINING BUDGET {budget}")
        print(f"======== ITERATION {iteration}/{num_iterations}")
        writer.add_scalar("info/iteration", iteration, global_step)

        for step in range(0, train_params.num_steps):
            step_start_time = time.time()
            global_step += train_params.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute the game
            next_obs, r, next_done, info = envs.step(action)
            with torch.no_grad():
                rewards[step] = reward_normalizer(r) if train_params.norm_reward else r

            if any(next_done):
                done_mean_ep_returns.append(info["r"][next_done == 1].mean().item())
                done_mean_ep_len.append(info["l"][next_done == 1].float().mean().item())
                done_ep_cses.append(info["consecutive_successes"].item())

        # Log the data
        episodic_length = np.mean(done_mean_ep_len)
        writer.add_scalar("episode_lengths", episodic_length, global_step)
        episodic_returns = np.mean(done_mean_ep_returns)
        writer.add_scalar("episodic_returns", episodic_returns, global_step)

        writer.add_scalar("unnormalized_rewards", r.mean().item(), global_step)
        writer.add_scalar("rewards", rewards[step].mean().item(), global_step)

        cs = info["consecutive_successes"].item()

        writer.add_scalar("consecutive_successes", cs, global_step)

        if "gt_reward" in info and "gpt_reward" in info:
            writer.add_scalar("rewards/gt_reward", info["gt_reward"].item(), global_step)
            writer.add_scalar("rewards/gpt_reward", info["gpt_reward"].item(), global_step)

        # Add reward components
        for k, v in envs.rew_dict.items():
            if k not in keys_to_ignore:
                writer.add_scalar(f"components/{k}", v.mean().item(), global_step)

        sps = int(global_step / (time.time() - start_time))
        immediate_sps = int(train_params.num_envs / (time.time() - step_start_time))
        spe = int(global_step / train_params.num_envs)
        
        print_log(f"global_step: {global_step:,.0f} | steps_per_env: {spe} | sps: {sps:,.0f} | immediate_sps: {immediate_sps:,.0f} | ep_return: {episodic_returns:.4f} | consecutive_successes: {cs}")

        # Bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(train_params.num_steps)):
                if t == train_params.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + train_params.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + train_params.gamma * train_params.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(train_params.update_epochs):
            b_inds = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > train_params.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if train_params.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - train_params.clip_coef, 1 + train_params.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if train_params.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -train_params.clip_coef,
                        train_params.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - train_params.ent_coef * entropy_loss + v_loss * train_params.vf_coef

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if train_params.truncate_grads:
                    nn.utils.clip_grad_norm_(agent.parameters(), train_params.max_grad_norm)
                optimizer.step()

                if train_params.lr_schedule == "adaptive":
                    optimizer.param_groups[0]["lr"] = scheduler.update(optimizer.param_groups[0]["lr"], approx_kl)

        writer.add_scalar("info/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("info/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("info/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("info/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("info/steps_per_second", sps, global_step)
        writer.add_scalar("info/immediate_steps_per_second", immediate_sps, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

        # Check if it's time to save a checkpoint
        # if iteration % train_params.checkpoint_freq == 0:
        #     # Save checkpoint
        #     checkpoint_path = os.path.join(f"{log_dir}/runs/{run_name}/nn", f"checkpoint_iter_{iteration}_cs_{cs:.4f}.pth")
        #     torch.save({
        #         'iteration': iteration,
        #         'model_state_dict': agent.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #     }, checkpoint_path)
        #     print(f"=> saving checkpoint '{checkpoint_path}'")

        # Check if the performance of the current policy exceeds the best performance
        if cs > best_cs and iteration > train_params.save_best_after:
            best_cs = cs
            best_checkpoint_path = os.path.join(f"{log_dir}/runs/{run_name}/nn", f"best_checkpoint.pth")
            torch.save({
                'iteration': iteration,
                'model_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_checkpoint_path)
            print(f"=> saving checkpoint '{best_checkpoint_path}'")

        if budget < 0:
            print("BUDGET EXHAUSTED")
            break
        if iteration != num_iterations:
            budget -= 1

    print(f"REMAINING BUDGET: {budget}")

    if cfg.capture_video:
        envs.close()
    writer.close()

    # Collect statistics
    tensorboard_data = load_tensorboard_logs(f"{log_dir}/runs/{run_name}/summaries")

    consecutive_successes = np.array(tensorboard_data["consecutive_successes"])

    # Save npz file with the statistics
    npz_file = f"{log_dir}/runs/{run_name}/training_statistics.npz"
    np.savez(npz_file, consecutive_successes=consecutive_successes)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(consecutive_successes)
    
    ax.set_title("Task Score", weight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Task Score")

    fig.savefig(f"{log_dir}/task_score_last_iter.pdf", bbox_inches='tight')
    fig.savefig(f"{log_dir}/task_score_last_iter.png", bbox_inches='tight')

if __name__ == "__main__":
    main()
