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

import argparse
import math

import gym
import isaacgymenvs
import torch
from torch.distributions.normal import Normal
from utils.misc import *
from utils.wrappers import ExtractObsWrapper, RecordEpisodeStatisticsTorch

import isaacgym  # noqa


def main():

    # Seeding
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="check_ant")
    parser.add_argument("--suffix", type=str, default="GPT")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50)
    args = parser.parse_args()

    target_name = args.env
    num_envs = args.num_envs
    num_steps = args.num_steps

    target_task = ''.join([word.capitalize() for word in target_name.split('_')]) + args.suffix

    keys_to_ignore = ["consecutive_successes", "time_outs", "r", "l", "gt_reward", "gpt_reward"]
    freq = int(num_steps / 10)

    set_freest_gpu()

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = isaacgymenvs.make(
        seed=42,
        task=target_task,
        num_envs=num_envs,
        sim_device="cpu", # "cuda:0" if torch.cuda.is_available() else "cpu",
        rl_device="cpu", # "cuda:0" if torch.cuda.is_available() else "cpu",
        graphics_device_id=-1, # 0 if torch.cuda.is_available() else -1,
        headless=True,
        multi_gpu=False,
        virtual_screen_capture=False,
        force_render=False,
    )
    envs = ExtractObsWrapper(envs)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    envs.reset()

    statistics = dict()

    for step in range(1, num_steps + 1):

        # Sample random actions
        action = torch.tensor(np.array([envs.action_space.sample() for _ in range(num_envs)])).to(device)

        _, rewards, _, info = envs.step(action)

        # Record statistics
        step_rewards = rewards.mean().item()
        print(f"======== STEP {step}/{num_steps} | REWARD {step_rewards}")
        if "reward" not in statistics:
            statistics["reward"] = [step_rewards]
        else:
            statistics["reward"].append(step_rewards)

        for k, v in info.items():
            if k not in keys_to_ignore:
                if k not in statistics:
                    statistics[k] = [v.mean().item()]
                else:
                    statistics[k].append(v.mean().item())

    # Check statistics
    stats_string = ""
    is_nan = is_inf = is_neg_inf = False
    for k, v in statistics.items():
        if np.nan in v or math.nan in v:
            print(f"NaN found in {k}")
            is_nan = True
        if np.inf in v:
            print(f"Inf found in {k}")
            is_inf = True
        if -np.inf in v:
            print(f"-Inf found in {k}")
            is_neg_inf = True
        stats_string += f"{k}: {[round(val, 4) for val in v[::freq]]}\n"

    traceback = ""
    if is_nan:
        traceback += "The reward or some of its components are NaN. "
    if is_inf:
        traceback += "The reward or some of its components are Inf. "
    if is_neg_inf:
        traceback += "The reward or some of its components are -Inf. "

    if traceback != "":
        traceback += "\n" + stats_string
        traceback = target_task + ":\n" + traceback
        raise RuntimeError(traceback)

    print("All checks passed!")
    print(f"Statistics:\n{stats_string}")

if __name__ == "__main__":
    main()
