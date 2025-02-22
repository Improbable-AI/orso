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
import importlib
import json
import logging
import os
import random
import sys
import time
from collections import deque

import colorcet as cc
import gym
import hydra
import isaacgymenvs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from agent import Agent
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.agent_utils import RunningMeanStd
from utils.generation import *
from utils.messages import *
from utils.misc import *
from utils.prune_env import prune
from utils.schedulers import AdaptiveScheduler
from utils.wrappers import ExtractObsWrapper, RecordEpisodeStatisticsTorch

import isaacgym  # noqa

SRC_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{SRC_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
LOG_ROOT_DIR = os.getenv("LOG_ROOT_DIR") + "/orso" if os.getenv("LOG_ROOT_DIR") else SRC_ROOT_DIR

np.set_printoptions(linewidth=np.inf)


def plot(r_idxes, cs_list, colors, log_dir, max_successes=None):

    sns.set_theme(
        context='paper',
        style="whitegrid",
        rc={"lines.linewidth": 1},
    )

    # Plot consecutive successes
    fig, ax = plt.subplots(figsize=(10, 5))

    prev_color = None
    x_tick = 0
    for i, (r_indices, r_values) in enumerate(zip(r_idxes, cs_list)):
        for r_i_idx, r_i_values in zip(r_indices, r_values):
            color = colors[r_i_idx]  # Get color from color palette

            if prev_color is not None:
                ax.plot([x_tick - 1, x_tick], [prev_val, r_i_values[0]], color=color)

            x_ticks = np.arange(x_tick, x_tick + len(r_i_values))
            ax.plot(x_ticks, r_i_values, color=color)

            nan_indices = np.isnan(r_i_values)
            x_ticks_nan = np.arange(x_tick, x_tick + len(r_i_values))[nan_indices]
            zero_vec = [0 for _ in np.array(r_i_values)[nan_indices]]
            ax.plot(x_ticks_nan, zero_vec, marker="X", color=color, linestyle="None", zorder=10, markersize=10)

            prev_val = r_i_values[-1]
            prev_color = color

            x_tick += len(r_i_values)  # Increment x value for next r_i

        # Draw vertical line after each evo_i
        ax.axvline(x=x_tick - 1, color="k", linestyle="--")

    ax.set_title("Task Score", weight="bold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Task Score")
    # ax.grid(True, which='major', linewidth=0.5)
    # ax.spines[:].set_visible(False)

    fig.savefig(f"{log_dir}/fitness.pdf", bbox_inches="tight")
    fig.savefig(f"{log_dir}/fitness.png", bbox_inches="tight")

    if max_successes is not None:

        # Plot max successes
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(max_successes)

        ax.set_title("Max Task Score", weight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Task Score")
        # ax.grid(True, which='major', linewidth=0.5)
        # ax.spines[:].set_visible(False)

        fig.savefig(f"{log_dir}/max_fitness.pdf", bbox_inches="tight")
        fig.savefig(f"{log_dir}/max_fitness.png", bbox_inches="tight")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    overall_start_time = time.time()

    target_task = cfg.env.task  # ShadowHand
    target_name = cfg.env.name  # shadow_hand
    target_description = cfg.env.description

    network_params = cfg.env.network
    train_params = cfg.env.train
    mab_params = cfg.env.mab

    assert mab_params.n_arms > 1, "n_arms must be strictly greater than 1"

    llm_params = {"model": cfg.model, "temperature": cfg.temperature, "samples": mab_params.n_arms}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    ###########################################################################
    # ================================ LOGGING ================================
    ###########################################################################

    # Create logging directory
    os.makedirs(f"{LOG_ROOT_DIR}/logs", exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"orso_{mab_params.algo}_{cfg.env.task}_b_{cfg.budget}_k_{mab_params.n_arms}_{timestamp}"

    os.makedirs(f"{SRC_ROOT_DIR}/envs/{timestamp}")

    log_dir = f"{LOG_ROOT_DIR}/logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{log_dir}/train_output.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )

    print_log(f"Log directory: {log_dir}")

    # Print configuration
    print_log("======== Configuration ========")
    print_log(OmegaConf.to_yaml(cfg))
    print_log("=" * len("======== Configuration ========"))

    print_log("======== Task ========")
    print_log(f"Task: {target_task}")
    print_log(f"Name: {target_name}")
    print_log(f"Description: {target_description}")
    print_log("=" * len("======== Task ========"))

    wandb_tags = [mab_params.algo, f"b_{cfg.budget}", f"k_{mab_params.n_arms}"]
    wandb_group = f"{target_task}_{mab_params.algo}_b_{cfg.budget}_k_{mab_params.n_arms}"

    if cfg.use_wandb:
        import wandb

        wandb_unique_id = f"uid_{run_name}"
        print_log(f"WandB using unique id {wandb_unique_id}")

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
    os.makedirs(f"{log_dir}/runs/{run_name}/nn", exist_ok=True)

    ###########################################################################
    # =========================== PRUNE ENVIRONMENT ===========================
    ###########################################################################

    # Prune the env
    if "hand" in target_name:
        prune(task=target_name, methods_to_keep=["__init__"])
    else:
        prune(task=target_name)
    pruned_code = file_to_string(f"{SRC_ROOT_DIR}/envs/{target_name}.py")
    pruned_obs_code = file_to_string(f"{SRC_ROOT_DIR}/envs/{target_name}_obs.py")

    # Make YAML for train
    original_train_config = file_to_string(f"{ISAAC_ROOT_DIR}/cfg/train/{target_task}PPO.yaml")
    with open(f"{ISAAC_ROOT_DIR}/cfg/train/{target_task}{cfg.suffix}PPO.yaml", "w") as file:
        new_yaml = original_train_config.replace(target_task, f"{target_task}{cfg.suffix}")
        file.write(new_yaml)
    # Make YAML file for task
    original_task_config = file_to_string(f"{ISAAC_ROOT_DIR}/cfg/task/{target_task}.yaml")
    with open(f"{ISAAC_ROOT_DIR}/cfg/task/{target_task}{cfg.suffix}.yaml", 'w') as file:
        file.write(original_task_config.replace(target_task, f"{target_task}{cfg.suffix}"))

    # Make a copy of the environment file and change the class name
    with open(f"{ISAAC_ROOT_DIR}/tasks/{target_name}{cfg.suffix}.py", "w") as file:
        file.write(pruned_code.replace(target_task, f"{target_task}{cfg.suffix}"))

    # Make a file to store previous tracebacks
    with open(f"{log_dir}/_prev_tracebacks.txt", "w") as f:
        f.write("")

    ###########################################################################
    # =========================== MAKE ENVIRONMENT ============================
    ###########################################################################

    gpus = set_freest_gpu()

    while True:
        try:
            envs = isaacgymenvs.make(
                seed=cfg.seed,
                task=f"{target_task}{cfg.suffix}",
                num_envs=train_params.num_envs,
                sim_device=f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu",
                rl_device=f"cuda:{gpus[0]}" if torch.cuda.is_available() else "cpu",
                graphics_device_id=int(gpus[0]) if torch.cuda.is_available() else -1,
                headless=True,
                multi_gpu=False,
                virtual_screen_capture=False,
                force_render=False,
            )
            break
        except Exception as e:  # This is so that if some invalid codes are created, I retry until all are valid (from eg another task)
            print_log(f"Failed to make environment. Trying again. Error: {e}")
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

    if train_params.lr_schedule == "adaptive":
        scheduler = AdaptiveScheduler(kl_threshold=train_params.kl_threshold)

    scaler = torch.cuda.amp.GradScaler(enabled=train_params.mixed_precision)

    ###########################################################################
    # ============================= PPO VARIABLES =============================
    ###########################################################################

    # Storage variables
    obs = torch.zeros((train_params.num_steps, train_params.num_envs) + envs.single_observation_space.shape, dtype=torch.float).to(device)
    actions = torch.zeros((train_params.num_steps, train_params.num_envs) + envs.single_action_space.shape, dtype=torch.float).to(device)
    logprobs = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    values = torch.zeros((train_params.num_steps, train_params.num_envs), dtype=torch.float).to(device)
    advantages = torch.zeros_like(rewards, dtype=torch.float).to(device)

    # PPO parameters
    batch_size = int(train_params.num_envs * train_params.num_steps)
    minibatch_size = int(batch_size // train_params.num_minibatches)
    num_iterations = train_params.total_timesteps // batch_size
    keys_to_ignore = ["consecutive_successes", "time_outs", "r", "l"]

    print_log(f"===============================")
    print_log(f"Batch size: {batch_size}")
    print_log(f"Minibatch size: {minibatch_size}")
    print_log(f"Number of iterations: {num_iterations}")

    mab_params.update_freq = max(1, int(num_iterations // 100))
    print_log(f"Update frequency: {mab_params.update_freq}")

    total_budget = cfg.budget * num_iterations
    print_log(f"Total budget: {total_budget}")

    ###########################################################################
    # ============================ TRAINING LOOP =============================
    ###########################################################################

    rewards_indices = np.arange(mab_params.n_arms)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", 
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"
    ]
    colors = cc.palette['cyclic_rygcbmr_50_90_c64'][::len(cc.palette['cyclic_rygcbmr_50_90_c64'])//mab_params.n_arms]

    cs_list = []
    r_idxes = []

    freqs = []

    init_obs = envs.reset()
    init_done = torch.zeros(train_params.num_envs, dtype=torch.float).to(device)
    init_env_vars = envs.get_all_env_vars()

    prev_obs = [None for _ in range(mab_params.n_arms)]
    prev_done = [None for _ in range(mab_params.n_arms)]
    previous_env_vars = [dict() for _ in range(mab_params.n_arms)]

    best_cs = -np.inf  # best consecutive successes
    max_successes = []
    best_idxes = []

    # Start the game
    evo = 0
    global_step = 0

    b = 1

    start_time = time.time()

    while b < total_budget:

        #######################################################################
        # ========================== GENERATE REWARDS =========================
        #######################################################################

        if evo == 0:
            finished = False
            messages = init_messages(pruned_obs_code, target_description)

            print_log("=" * 80)
            print_log("GENERATING INITIAL REWARD FUNCTIONS.")
            print_log("=" * 80)

            responses, _ = generate_rewards(log_dir, llm_params, pruned_code, messages, target_task, cfg.suffix, cfg.n_attempts, mab_params.n_arms, rewards_indices, evo=evo, timestamp=timestamp)
        else:
            print_log("=" * 80)
            print_log(f"BEST REWARD FUNCTION {best_idx} WITH STATISTICS:\n{best_stats_string}")
            print_log("=" * 80)

            idxes_from_scratch = rewards_indices[:len(rewards_indices) // 2]
            idxes_not_from_scratch = rewards_indices[len(rewards_indices) // 2:]

            print_log(f"GENERATING REWARDS FOR {idxes_from_scratch} FROM SCRATCH")
            initial_messages = init_messages(pruned_obs_code, target_description)
            responses_scratch, _ = generate_rewards(log_dir, llm_params, pruned_code, initial_messages, target_task, cfg.suffix, cfg.n_attempts, len(idxes_from_scratch), idxes_from_scratch, evo, timestamp=timestamp)

            print_log(f"GENERATING REWARDS FOR {idxes_not_from_scratch} NOT FROM SCRATCH")
            if finished:
                print_log("FINISHED :)")
                messages = update_messages(log_dir, messages, responses[best_idx], best_stats_string, freq)
                finished = False
            else:
                print_log("NOT FINISHED :(")
            responses_not_scratch, _ = generate_rewards(log_dir, llm_params, pruned_code, messages, target_task, cfg.suffix, cfg.n_attempts, len(idxes_not_from_scratch), idxes_not_from_scratch, evo, timestamp=timestamp)

            responses = responses_scratch + responses_not_scratch

        #######################################################################
        # ========================= BANDIT VARIABLES ==========================
        #######################################################################
            
        current_idx = random.choice(rewards_indices)

        bandit_rewards = np.array([100_000.0] * len(rewards_indices))
        rewards_freq = np.zeros(len(rewards_indices))  # counts how many times each arm was chosen
        iters_per_arm = np.zeros(len(rewards_indices))
        iteration = 1

        cons_succ_list = [[] for _ in range(len(rewards_indices))]  # consecutive successes for each arm
        components_list = [dict() for _ in range(len(rewards_indices))]  # reward components for each arm

        prev_cses = np.zeros(len(rewards_indices))  # previous consecutive successes for each arm
        best_cses = np.array([-np.inf] * len(rewards_indices))

        if mab_params.algo == "exp3":
            weights = np.ones(len(rewards_indices))
            probs = weights / weights.sum()
            reward_estimates = np.zeros(len(rewards_indices))

        if mab_params.algo == "d3rb" or mab_params.algo == "ed2rb":
            min_regret_coefficient = 1.0
            failure_prob = 0.1
            balancing_potentials = np.array([min_regret_coefficient] * len(rewards_indices))
            regret_coeffs = np.array([min_regret_coefficient] * len(rewards_indices))

        agents = [Agent(envs, network_params, train_params.norm_input).to(device) for _ in range(mab_params.n_arms)]
        optimizers = [optim.Adam(agent.parameters(), lr=train_params.learning_rate, eps=1e-5) for agent in agents]

        reward_normalizers = nn.ModuleList([RunningMeanStd((1,)) for _ in range(mab_params.n_arms)]).to(device)

        print_log(f"USING REWARD FUNCTION {current_idx}")
        imported_module = importlib.import_module(f'src.envs.{timestamp}.{target_name}{cfg.suffix}_{evo}_{current_idx}')
        method = getattr(imported_module, 'compute_reward')
        replace_method(f"isaacgymenvs.tasks.{target_name}{cfg.suffix}", f"{target_task}{cfg.suffix}", 'compute_reward', method)

        print_log(f"USING NETWORK {current_idx}")
        agent = agents[current_idx]
        optimizer = optimizers[current_idx]

        next_obs = init_obs.clone()
        next_done = init_done.clone()
        envs.set_all_env_vars(init_env_vars)

        cs_list.append([])
        r_idxes.append([])

        tmp_cs_list = []
        bandit_step_cs_list = []
        reg = np.zeros(mab_params.n_arms)

        done_mean_ep_returns = [deque(maxlen=train_params.num_steps) for _ in range(mab_params.n_arms)]
        done_mean_ep_len = [deque(maxlen=train_params.num_steps) for _ in range(mab_params.n_arms)]
        done_ep_cses = [deque(maxlen=train_params.num_steps) for _ in range(mab_params.n_arms)]

        while np.max(iters_per_arm) < num_iterations:
            print_log(f"======== ITERATION {iteration} | EVOLUTION {evo}")
            writer.add_scalar("info/iteration", iteration, global_step)
            writer.add_scalar("info/evolution", evo, global_step)

            iters_per_arm[current_idx] += 1
            print_log(f"iters_per_arm: {iters_per_arm} / {num_iterations} | budget: {b} / {total_budget}")

            ###################################################################
            # ============================ ROLLOUT ============================
            ###################################################################

            invalid_rewards = False

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
                try:
                    next_obs, r, next_done, info = envs.step(action)
                except Exception as e:
                    print_log("!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!")
                    print_log(e)
                    invalid_rewards = True
                    
                    next_obs = init_obs.clone()
                    r = torch.zeros(train_params.num_envs, dtype=torch.float).to(device)
                    next_done = init_done.clone()
                    info = {
                        "r": torch.zeros(train_params.num_envs, dtype=torch.float).to(device),
                        "l": torch.zeros(train_params.num_envs, dtype=torch.float).to(device),
                        "consecutive_successes": torch.zeros(train_params.num_envs, dtype=torch.float).to(device),
                    }

                with torch.no_grad():
                    rewards[step] = reward_normalizers[current_idx](r) if train_params.norm_reward else r
                    
                # Compute the mean of the episodic returns for the done envs
                if any(next_done):
                    done_mean_ep_returns[current_idx].append(info["r"][next_done == 1].mean().item())
                    done_mean_ep_len[current_idx].append(info["l"][next_done == 1].float().mean().item())
                    done_ep_cses[current_idx].append(info["consecutive_successes"].item())

            ###################################################################
            # ============================ LOGGING ============================
            ###################################################################

            # Log the data
            if len(done_mean_ep_returns[current_idx]) == 0:
                done_mean_ep_returns[current_idx].append(0.0)
            if len(done_mean_ep_len[current_idx]) == 0:
                done_mean_ep_len[current_idx].append(0.0)
            if len(done_ep_cses[current_idx]) == 0:
                done_ep_cses[current_idx].append(0.0)

            episodic_length = np.mean(done_mean_ep_len[current_idx])
            writer.add_scalar("episode_lengths", episodic_length, global_step)
            episodic_returns = np.mean(done_mean_ep_returns[current_idx])
            writer.add_scalar("episodic_returns", episodic_returns, global_step)

            writer.add_scalar("unnormalized_rewards", r.mean().item(), global_step)
            writer.add_scalar("rewards", rewards[step].mean().item(), global_step)

            cs = -np.inf if invalid_rewards else np.mean(done_ep_cses[current_idx])
            cons_succ_list[current_idx].append(cs)
            bandit_step_cs_list.append(cs)
            tmp_cs_list.append(cs if cs != -np.inf else np.nan)
            
            writer.add_scalar("consecutive_successes", cs, global_step)

            if "gt_reward" in info and "gpt_reward" in info:
                writer.add_scalar("rewards/gt_reward", info["gt_reward"].item(), global_step)
                writer.add_scalar("rewards/gpt_reward", info["gpt_reward"].item(), global_step)

            if "episode_lengths" not in components_list[current_idx]:
                components_list[current_idx]["episode_lengths"] = [episodic_length]
            else:
                components_list[current_idx]["episode_lengths"].append(episodic_length)

            if hasattr(envs, "rew_dict"):

                print_log(f"REWARD COMPONENTS FOR ARM {current_idx}: {[k for k in envs.rew_dict.keys() if k not in keys_to_ignore]}")

                # Add reward components
                for k, v in envs.rew_dict.items():
                    if k not in keys_to_ignore:
                        try:
                            writer.add_scalar(f"components/{k}", v.mean().item(), global_step)
                            if k not in components_list[current_idx]:
                                components_list[current_idx][k] = [v.mean().item()]
                            else:
                                components_list[current_idx][k].append(v.mean().item())
                        except Exception as e:
                            print_log(f"Error logging reward component!")

            sps = int(global_step / (time.time() - start_time))
            immediate_sps = int(train_params.num_envs / (time.time() - step_start_time))
            spe = int(global_step / train_params.num_envs)

            print_log(f"global_step: {global_step:,.0f} | steps_per_env: {spe} | sps: {sps:,.0f} | immediate_sps: {immediate_sps:,.0f} | ep_return: {episodic_returns:.4f} | consecutive_successes: {cs}")

            ###################################################################
            # ========================= PPO TRAINING ==========================
            ###################################################################

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

            ###################################################################
            # ============================ LOGGING ============================
            ###################################################################

            writer.add_scalar("info/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("info/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("info/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("info/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("info/steps_per_second", sps, global_step)
            writer.add_scalar("info/immediate_steps_per_second", immediate_sps, global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)

            ###################################################################
            # ========================== CHECKPOINT ===========================
            ###################################################################

            # Check if it's time to save a checkpoint
            # if iteration % train_params.checkpoint_freq == 0:
            #     # Save checkpoint
            #     checkpoint_path = os.path.join(f"{log_dir}/runs/{run_name}/nn", f"checkpoint_iter_evo_{evo}_{iteration}_cs_{cs:.4f}.pth")
            #     torch.save({
            #         'iteration': iteration,
            #         'model_state_dict': agent.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'reward_norm_state_dict': reward_normalizers[current_idx].state_dict(),
            #     }, checkpoint_path)
            #     print_log(f"=> saving checkpoint '{checkpoint_path}'")

            # Check if the performance of the current policy exceeds the best performance
            if cs > best_cs and iteration > train_params.save_best_after:
                best_cs = cs
                best_checkpoint_path = os.path.join(f"{log_dir}/runs/{run_name}/nn", f"best_checkpoint.pth")
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': agent.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'reward_norm_state_dict': reward_normalizers[current_idx].state_dict(),
                }, best_checkpoint_path)
                print_log(f"=> saving checkpoint '{best_checkpoint_path}'")
            
            ###################################################################
            # ========================== MAB UPDATES ==========================
            ###################################################################
                
            if evo > 0:
                reg[current_idx] += (best_cs_list_so_far[int(iters_per_arm[current_idx])-1] - cs)

            # Update reward function and the current agent
            if iteration % mab_params.update_freq == 0 or invalid_rewards or np.max(iters_per_arm) == num_iterations or b == total_budget:

                prev_idx = current_idx

                wacs = np.sum(bandit_step_cs_list * np.power(mab_params.gamma, np.arange(len(bandit_step_cs_list))[::-1])) / np.sum(np.power(mab_params.gamma, np.arange(len(bandit_step_cs_list))[::-1]))

                print_log("=" * 80)
                print_log("=" * 80)

                # Delta and consecutive successes
                print_log(f"PREVIOUS WEIGHTED AVERAGE CONSECUTIVE SUCCESSES (WACS): {prev_cses}")
                print_log(f"PREVIOUS BEST WACS: {best_cses}")
                print_log(f"WACS of {current_idx}: {wacs}")

                prev_cses[current_idx] = wacs

                if wacs > best_cses[current_idx]:
                    best_cses[current_idx] = wacs

                # Compute bandit rewards
                bandit_rewards[current_idx] = (bandit_rewards[current_idx] * rewards_freq[current_idx] + wacs) / (rewards_freq[current_idx] + 1)
                print_log(f"BANDIT REWARDS: {bandit_rewards}")

                rewards_freq[current_idx] += 1
                freqs.append(rewards_freq.copy())
                print_log(f"REWARDS FREQUENCY: {rewards_freq}")

                # ======================== MAB ALGO LOGIC =========================
                if mab_params.algo == "etc":
                    if all(f == mab_params.n_exploration_updates for f in rewards_freq):
                        current_idx = np.argmax(bandit_rewards)
                    elif any([f < mab_params.n_exploration_updates for f in rewards_freq]):
                        current_idx = (current_idx + 1) % len(rewards_indices)

                elif mab_params.algo == "eg":
                    if random.random() > mab_params.epsilon or any([f == 0 for f in rewards_freq]):  # ensures all arms are explored
                        current_idx = np.argmax(bandit_rewards)
                    else:
                        current_idx = random.choice(rewards_indices)

                elif mab_params.algo == "ucb":
                    ci = np.sqrt(2.0 * np.log(iteration / float(mab_params.update_freq)) / rewards_freq)
                    ucb_values = bandit_rewards + mab_params.alpha * ci
                    print_log(f"UCB CONFIDENCE INTERVALS: {ci}")
                    print_log(f"UCB VALUES: {ucb_values}")
                    current_idx = np.argmax(ucb_values)

                # Pseudocode from "The Nonstochastic Multi-Armed Bandit Problem"
                elif mab_params.algo == "exp3":
                    reward_estimates[current_idx] = wacs / probs[current_idx]
                    print_log(f"REWARD ESTIMATES: {reward_estimates}")

                    # clamp exponentail part of the weights to avoid overflow
                    weights[current_idx] = np.clip(weights[current_idx] * np.exp(mab_params.eta * reward_estimates[current_idx] / mab_params.n_arms), 0, 1e100)
                    print_log(f"WEIGHTS: {weights}")

                    probs = (1 - mab_params.eta) * weights / np.sum(weights) + mab_params.eta / mab_params.n_arms
                    print_log(f"PROBS: {probs}")

                    if any([f == 0 for f in rewards_freq]):  # ensures all arms are explored
                        current_idx = (current_idx + 1) % len(rewards_indices)
                    else:
                        current_idx = np.random.choice(rewards_indices, p=probs)
                
                elif mab_params.algo == "d3rb":
                    hoeffding_bonuses = np.sqrt(np.log((mab_params.n_arms * np.log(rewards_freq)) / failure_prob) / rewards_freq)
                    lhs_misspecification_test = bandit_rewards[current_idx] + (regret_coeffs[current_idx] * np.sqrt(rewards_freq[current_idx])) / rewards_freq[current_idx] + mab_params.c * hoeffding_bonuses[current_idx]
                    rhs_mispecification_test = np.max(bandit_rewards - mab_params.c * hoeffding_bonuses)
                    if lhs_misspecification_test < rhs_mispecification_test:
                        regret_coeffs[current_idx] = regret_coeffs[current_idx] * 2.0
                    balancing_potentials[current_idx] = regret_coeffs[current_idx] * np.sqrt(rewards_freq[current_idx])

                    print_log(f"HOEFFDING BONUSES: {hoeffding_bonuses}")
                    print_log(f"LHS MISSPECIFICATION TEST: {lhs_misspecification_test}")
                    print_log(f"RHS MISSPECIFICATION TEST: {rhs_mispecification_test}")
                    print_log(f"REGRET COEFFICIENTS: {regret_coeffs}")
                    print_log(f"BALANCING POTENTIALS: {balancing_potentials}")

                    if any([f == 0 for f in rewards_freq]):  # ensures all arms are explored
                        current_idx = (current_idx + 1) % len(rewards_indices)
                    else:
                        current_idx = np.random.choice(np.where(balancing_potentials == np.min(balancing_potentials))[0])  # random tie breaking

                elif mab_params.algo == "ed2rb":
                    hoeffding_bonuses = np.sqrt(np.log((mab_params.n_arms * np.log(rewards_freq)) / failure_prob) / rewards_freq)
                    first_term = np.max(bandit_rewards - mab_params.c * hoeffding_bonuses)
                    second_term = bandit_rewards[current_idx] + mab_params.c * hoeffding_bonuses[current_idx]
                    regret_coeffs[current_idx] = max(min_regret_coefficient, np.sqrt(rewards_freq[current_idx]) * (first_term - second_term))
                    balancing_potentials[current_idx] = np.clip(regret_coeffs[current_idx] * np.sqrt(rewards_freq[current_idx]), balancing_potentials[current_idx], 2 * balancing_potentials[current_idx])

                    print_log(f"HOEFFDING BONUSES: {hoeffding_bonuses}")
                    print_log(f"FIRST TERM: {first_term}")
                    print_log(f"SECOND TERM: {second_term}")
                    print_log(f"REGRET COEFFICIENTS: {regret_coeffs}")
                    print_log(f"BALANCING POTENTIALS: {balancing_potentials}")

                    if any([f == 0 for f in rewards_freq]):  # ensures all arms are explored
                        current_idx = (current_idx + 1) % len(rewards_indices)
                    else:
                        current_idx = np.random.choice(np.where(balancing_potentials == np.min(balancing_potentials))[0])

                else:
                    raise NotImplementedError

                # ====================== SET REWARD FUNCTION ======================

                if current_idx != prev_idx:
                    
                    prev_obs[prev_idx] = next_obs
                    prev_done[prev_idx] = next_done
                    previous_env_vars[prev_idx] = envs.get_all_env_vars()
                    
                    print_log(f"USING REWARD FUNCTION {current_idx}")
                    imported_module = importlib.import_module(f'src.envs.{timestamp}.{target_name}{cfg.suffix}_{evo}_{current_idx}')
                    method = getattr(imported_module, 'compute_reward')
                    replace_method(f"isaacgymenvs.tasks.{target_name}{cfg.suffix}", f"{target_task}{cfg.suffix}", 'compute_reward', method)

                    print_log(f"USING NETWORK {current_idx}")
                    agent = agents[current_idx]
                    optimizer = optimizers[current_idx]

                    if all(f > 0 for f in rewards_freq):
                        print_log("LOADING PREVIOUS ENVIRONMENT VARIABLES")
                        next_obs = prev_obs[current_idx]
                        next_done = prev_done[current_idx]
                        envs.set_all_env_vars(previous_env_vars[current_idx])
                    else:
                        next_obs = init_obs.clone()
                        next_done = init_done.clone()
                        envs.set_all_env_vars(init_env_vars)

                    cs_list[-1].append(tmp_cs_list)
                    r_idxes[-1].append(prev_idx)
                    tmp_cs_list = []
                    bandit_step_cs_list = []
                else:
                    if r_idxes[-1][-1] != current_idx:
                        cs_list[-1].append(tmp_cs_list)
                        r_idxes[-1].append(current_idx)
                    else:
                        cs_list[-1][-1].extend(tmp_cs_list)
                    tmp_cs_list = []
                    bandit_step_cs_list = []

            if evo > 0:
                print_log(f"reg: {reg}")
                print_log(f"avg_reg test: {reg / iters_per_arm} > {ms * np.sqrt(iters_per_arm) / iters_per_arm}?")

                # most_visited_r = np.argmax(iters_per_arm)

                # # if there is more than one arm with the same number of iterations, choose the one with the lowest regret
                # if np.sum(iters_per_arm == iters_per_arm[most_visited_r]) > 1:
                #     # the following screws up the indices, if reg = [1, 1, 0, 1] and iters_per_arm = [1, 3, 3, 2], I want most_visited_r = 2, but the result is 1
                #     # most_visited_r = np.argmin(reg[iters_per_arm == iters_per_arm[most_visited_r]])

                # most_visited_indices = np.where(iters_per_arm >= np.max(iters_per_arm) - mab_params.update_freq)[0]
                # most_visited_r = most_visited_indices[np.argmin(reg[most_visited_indices])]
                
                if np.all(reg / iters_per_arm > ms * np.sqrt(iters_per_arm) / iters_per_arm) and ms > max(max(sublist) for sublist in cons_succ_list):
                # if reg[most_visited_r] / iters_per_arm[most_visited_r] > 2 * ms * np.sqrt(iters_per_arm[most_visited_r]) / iters_per_arm[most_visited_r]:
                    print_log("REGRET IS TOO HIGH!!!!!!!!")
                    print_log("=" * 80)
                    break

            if iteration % 100 == 0:
                plot(r_idxes, cs_list, colors, log_dir)

            iteration += 1
            b += 1

            if b > total_budget:
                print_log("TOTAL BUDGET REACHED")
                print_log("=" * 80)
                break
            print_log("=" * 80)

        #######################################################################
        # ============================= EVOLUTION =============================
        #######################################################################

        total_time = time.time() - overall_start_time
        print_log(f"================= TOTAL TIME: {total_time} s")
            
        if len(tmp_cs_list) > 0:
            cs_list[-1].append(tmp_cs_list)
            r_idxes[-1].append(current_idx)

        best_idx = np.argmax(best_cses)
        best_idxes.append(best_idx)

        # Make stats string for the best reward function
        best_cs_list = cons_succ_list[best_idx]
        if len(best_cs_list) == num_iterations:
            finished = True
        best_components = components_list[best_idx]
        freq = int(len(best_cs_list) / 10) if len(best_cs_list) > 10 else 1
        best_stats_string = f"success: {[round(val, 4) for val in best_cs_list[::freq]]}, max: {round(np.max(best_cs_list), 4)}, min: {round(np.min(best_cs_list), 4)}, mean: {round(np.mean(best_cs_list), 4)}\n"
        for k, v in best_components.items():
            best_stats_string += f"{k}: {[round(val, 4) for val in v[::freq]]}, max: {round(np.max(v), 4)}, min: {round(np.min(v), 4)}, mean: {round(np.mean(v), 4)}\n"

        max_successes.append(np.max(best_cs_list))

        if np.argmax(max_successes) == evo:
            print_log("NEW BEST CONSECUTIVE SUCCESSES")
            best_cs_list_so_far = best_cs_list.copy()
            ms = np.max(max_successes[-1])

            if len(best_cs_list) != num_iterations:
                # Pad the list with the last value
                best_cs_list_so_far.extend([best_cs_list_so_far[-1]] * (num_iterations - len(best_cs_list)))

        # Plot results so far
        plot(r_idxes, cs_list, colors, log_dir, max_successes)

        # Save npz file with the statistics
        npz_file = f"{log_dir}/orso_{mab_params.algo}_training_statistics.npz"
        np.savez(npz_file, consecutive_successes=cs_list, r_idxes=r_idxes, max_successes=max_successes, total_time=total_time, freqs=freqs)

        with open(f'{log_dir}/messages.json', 'w') as file:
            json.dump(messages, file, indent=4)

        evo += 1

    if cfg.capture_video:
        envs.close()
    writer.close()

    for i, idx in enumerate(best_idxes):
        print_log(f"================= BEST EVO {i} ARM IS {idx}")
        print_log(f"================= MAX CS: {max_successes[i]}")
        print_log(f"================= BEST PATH: {log_dir}/{target_name}{cfg.suffix}_{i}_{idx}.py")
        print_log(f"================= BEST FULL PATH: {log_dir}/{target_name}{cfg.suffix}_{i}_{idx}_full.py")

    print_log("=" * 80)

    print_log("=" * 80)
    print_log(f"================= FINISHED ORSO ==================")
    print_log("=" * 80)

    total_time = time.time() - overall_start_time
    print_log(f"================= TOTAL TIME: {total_time} s")

    print_log("=" * 80)

if __name__ == "__main__":
    main()
