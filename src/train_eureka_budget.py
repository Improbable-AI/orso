# Code based on https://github.com/eureka-research/Eureka/blob/main/eureka/eureka.py

import datetime
import json
import logging
import os
import shutil
import subprocess
import time

import hydra
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from utils.generation import sample_llm
from utils.messages import *
from utils.misc import *
from utils.prune_env import *

SRC_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{SRC_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"
LOG_ROOT_DIR = os.getenv("LOG_ROOT_DIR") + "/orso" if os.getenv("LOG_ROOT_DIR") else SRC_ROOT_DIR

sns.set_theme(
    context="paper",
    style="whitegrid",
    rc={
        "lines.linewidth": 3,
    },
    font_scale=1.5,
)

np.set_printoptions(linewidth=np.inf)

DUMMY_FAILURE = -10000.

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    overall_start_time = time.time()

    target_task = cfg.env.task  # ShadowHand
    target_name = cfg.env.name  # shadow_hand
    target_description = cfg.env.description

    train_params = cfg.env.train

    # Create logging directory
    if not os.path.exists(f"{LOG_ROOT_DIR}/logs"):
        os.makedirs(f"{LOG_ROOT_DIR}/logs")
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    run_name = f"eureka_{target_task}_b_{cfg.budget}_k_{cfg.samples}_{timestamp}"
    log_dir = f"{LOG_ROOT_DIR}/logs/{run_name}"
    os.makedirs(log_dir, exist_ok=True)

    wandb_tags = ["eureka", f"b_{cfg.budget}", f"k_{cfg.samples}"]

    # Log outputs to a file
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

    # Prune the env
    if "hand" in target_name:
        prune_eureka(task=target_name, methods_to_keep=["__init__"])
    else:
        prune_eureka(task=target_name)
    pruned_code = file_to_string(f"{SRC_ROOT_DIR}/envs/{target_name}.py").replace(target_task, f"{target_task}{cfg.suffix}")
    pruned_obs_code = file_to_string(f"{SRC_ROOT_DIR}/envs/{target_name}_eureka_obs.py")

    output_file = f"{ISAAC_ROOT_DIR}/tasks/{target_name}{cfg.suffix}.py"
    prompt_dir = f"{SRC_ROOT_DIR}/prompts"

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
    with open(f"{output_file}", "w") as file:
        file.write(pruned_code)    

    # Prompts
    execution_error_feedback = file_to_string(f"{prompt_dir}/execution_error_feedback.txt")
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")

    # Initialize messages for the LLM
    messages = init_messages(pruned_obs_code, target_description)

    # Generate reward function with LLM with its corresponding env
    llm_params = {"model": cfg.model, "temperature": cfg.temperature, "samples": cfg.samples}

    max_successes = []
    cs_list = []
    unsuccsessful_count = 0
    
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", 
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"
    ]
    max_success_overall = DUMMY_FAILURE

    # PPO parameters
    batch_size = int(train_params.num_envs * train_params.num_steps)
    minibatch_size = int(batch_size // train_params.num_minibatches)
    num_iterations = train_params.total_timesteps // batch_size
    keys_to_ignore = ["consecutive_successes", "time_outs", "r", "l"]

    print_log(f"===============================")
    print_log(f"Batch size: {batch_size}")
    print_log(f"Minibatch size: {minibatch_size}")
    print_log(f"Number of iterations: {num_iterations}")

    total_budget = cfg.budget * num_iterations
    budget = total_budget
    print_log(f"Total budget: {total_budget}")

    iter = 0  # this is actually the evo
    
    while budget > 0:

        print_log("=" * len(f"EUREKA ITERATION {iter}") * 2)
        print_log(f"EUREKA ITERATION {iter}")
        print_log("=" * len(f"EUREKA ITERATION {iter}") * 2)

        # Get Eureka response
        print_log(f"Generating {llm_params['samples']} responses from LLM...")
        responses = sample_llm(
            model = llm_params["model"],
            messages = messages,
            temperature = llm_params["temperature"],
            n = llm_params["samples"],
        )
        print_log(f"Sampled {len(responses)} responses from LLM.")

        freest_gpus = find_freest_gpu()
        print_log(f"Freest GPUs: {freest_gpus}")
        
        code_runs = [] 
        rl_runs = []

        code_feedbacks = []
        contents = []
        successes = []
        code_paths = []
        cons_succ_list = []
        
        exec_success = False 

        for response_id, r in enumerate(responses):

            print_log(f"==== Iteration {iter} | Code Run {response_id} ====")

            code_string = parse_python_code(r)

            # Remove unnecessary imports
            lines = code_string.split("\n")
            for j, line in enumerate(lines):
                if line.strip().startswith("def "):
                    if line.strip().startswith("def compute_reward("):
                        lines[j] = lines[j].replace("def compute_reward(", "def compute_gpt_reward(")
                    code_string = "\n".join(lines[j:])
                    break
                    
            # Add the Eureka Reward Signature to the environment code
            try:
                gpt_reward_signature, input_lst = get_function_signature(code_string)
            except Exception as e:
                print_log(f"Iteration {iter}: Code Run {response_id} cannot parse function signature!")
                continue

            code_runs.append(code_string)
            reward_signature = [
                f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
            ]
            indent = " " * 8
            reward_signature = "\n".join([indent + line for line in reward_signature])
            if "def compute_reward(self)" in pruned_code:
                task_code_string_iter = pruned_code.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
            elif "def compute_reward(self, actions)" in pruned_code:
                task_code_string_iter = pruned_code.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
            else:
                raise NotImplementedError

            # Save the new environment code when the output contains valid code string!
            with open(output_file, 'w') as file:
                file.writelines(task_code_string_iter + '\n')
                file.writelines("from typing import Tuple, Dict" + '\n')
                file.writelines("import math" + '\n')
                file.writelines("import torch" + '\n')
                file.writelines("from torch import Tensor" + '\n')
                if "@torch.jit.script" not in code_string:
                    code_string = "@torch.jit.script\n" + code_string
                file.writelines(code_string + '\n')

            # Copy the generated environment code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"{log_dir}/env_iter{iter}_response{response_id}.py")
            
            # Execute the python file with flags
            rl_filepath = f"{log_dir}/env_iter{iter}_response{response_id}.txt"
            with open(rl_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{SRC_ROOT_DIR}/train_ppo_budget.py',
                                            f'env={target_name}', f'use_wandb={cfg.use_wandb}',
                                            f'wandb_username={cfg.wandb_username}', f'wandb_project={cfg.wandb_project}',
                                            f'capture_video={cfg.capture_video}', f'+log_dir={log_dir}', f'+remaining_budget={budget}',
                                            f'+wandb_group={target_task}_b_{cfg.budget}_k_{cfg.samples}_{timestamp}', f'+wandb_tags={wandb_tags}', f'seed={42+cfg.seed}'],
                                            stdout=f, stderr=f)

            rl_runs.append(process)
            process.wait()  # to run sequentially
        
            code_paths.append(f"{log_dir}/env_iter{iter}_response{response_id}.py")
            try:
                stdout_str = file_to_string(rl_filepath)
            except: 
                content = execution_error_feedback.format(traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!")
                content += code_output_tip
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                continue

            content = ''
            traceback_msg = filter_traceback(stdout_str)

            lines = stdout_str.split('\n')
            for line in lines[::-1]:
                if "REMAINING BUDGET: " in line:
                    budget = int(line.split("REMAINING BUDGET: ")[-1].strip())
                    break

            print_log(f"REMAINING BUDGET: {budget}/{total_budget}")

            if traceback_msg == '':
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                lines = stdout_str.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Tensorboard Directory:'):
                        break 
                tensorboard_logdir = line.split(':')[-1].strip() 
                tensorboard_logs = load_tensorboard_logs(tensorboard_logdir)
                max_iterations = np.array(tensorboard_logs['info/iteration']).shape[0]
                epoch_freq = max(int(max_iterations // 10), 1)
                
                content += policy_feedback.format(epoch_freq=epoch_freq)

                # Add reward components log to the feedback
                for metric in tensorboard_logs:
                    if "components/" in metric or metric == "consecutive_successes" or metric == "episode_lengths":

                        metric_cur = ['{:.2f}'.format(x) for x in tensorboard_logs[metric][::epoch_freq]]
                        metric_cur_max = max(tensorboard_logs[metric])
                        metric_cur_mean = sum(tensorboard_logs[metric]) / len(tensorboard_logs[metric])
                        metric_cur_min = min(tensorboard_logs[metric])

                        if metric == "consecutive_successes":
                            successes.append(metric_cur_max)
                            metric_name = "task_score"
                            cons_succ_list.append(tensorboard_logs[metric])
                        elif metric == "episode_lengths":
                            metric_name = metric
                        else:
                            metric_name = metric.split("/")[1]
                        content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"                    

                code_feedbacks.append(code_feedback)
                content += code_feedback  
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                cons_succ_list.append([DUMMY_FAILURE])
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 

            if budget < 0:
                print_log(f"BUDGET EXHAUSTED!")
                break
        
        print_log(f"==== Iteration {iter} Done ====")

        # Repeat the iteration if all code generation failed
        if not exec_success:
            unsuccsessful_count += 1
            print_log(f"Unsuccessful count: {unsuccsessful_count}")
            prev_unsuccessful = True
            print_log("All code generation failed! Repeat this iteration from the current message checkpoint!")
        else:
            unsuccsessful_count = 0
            print_log(f"Unsuccessful count: {unsuccsessful_count}")
            prev_unsuccessful = False

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
        max_success = successes[best_sample_idx]

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success

        max_successes.append(max_success)

        # Save all consecutive successes
        extension_list = [l if l != [DUMMY_FAILURE] else [np.nan] for l in cons_succ_list]
        cs_list.append(extension_list)

        print_log(f"Iteration {iter}: Max Success: {max_success}")
        print_log(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")

        # Save npz file with the statistics
        npz_file = f"{log_dir}/eureka_training_statistics.npz"
        np.savez(npz_file, consecutive_successes=cs_list, max_successes=max_successes)

        fig, ax = plt.subplots(figsize=(10, 5))

        # cs_list is in the format [evo_0, evo_1, ev_2, ...]
        # where evo_i = [r_1, r_2, r_3, r_4]
        # where r_i = [a, b, c, d, ...]
        prev_color = None
        x_tick = 0
        for evo_i in cs_list:
            for r_i, color in zip(evo_i, colors):
                if prev_color is not None:
                    plt.plot([x_tick - 1, x_tick], [prev_val, r_i[0]], color=color)

                x_ticks = np.arange(x_tick, x_tick + len(r_i))
                ax.plot(x_ticks, r_i, color=color)

                nan_indices = np.isnan(r_i)
                x_ticks_nan = np.arange(x_tick, x_tick + len(r_i))[nan_indices]
                zero_vec = [0 for _ in np.array(r_i)[nan_indices]]
                ax.plot(x_ticks_nan, zero_vec, marker='X', color=color, linestyle='None', zorder=10, markersize=10)

                prev_val = r_i[-1]
                prev_color = color

                x_tick += len(r_i)  # Increment x value for next r_i

            # Draw vertical line after each evo_i
            ax.axvline(x = x_tick-1, color='k', linestyle='--')
        
        ax.set_title("Task Score", weight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Task Score")

        fig.savefig(f"{log_dir}/task_score.pdf", bbox_inches='tight')
        fig.savefig(f"{log_dir}/task_score.png", bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(max_successes)
        
        ax.set_title("Max Task Score", weight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Task Score")

        fig.savefig(f"{log_dir}/max_task_score.pdf", bbox_inches='tight')
        fig.savefig(f"{log_dir}/max_task_score.png", bbox_inches='tight')

        if exec_success:
            if len(messages) == 2:
                messages += [{"role": "assistant", "content": responses[best_sample_idx]}]
                messages += [{"role": "user", "content": best_content}]
            else:
                assert len(messages) == 4
                messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]}
                messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open(f'{log_dir}/messages.json', 'w') as file:
            json.dump(messages, file, indent=4)

        iter += 1

        if unsuccsessful_count >= 10:
            print_log(f"Unsuccessful count reached 10! Exiting...")
            break

    print_log("=" * len("EUREKA FINISHED!") * 2)
    print_log("EUREKA FINISHED!")
    print_log("=" * len("EUREKA FINISHED!") * 2)

    # Print total time elapsed in seconds
    total_time = time.time() - overall_start_time
    print_log(f"================= TOTAL TIME: {total_time} s")

    npz_file = f"{log_dir}/eureka_training_statistics.npz"
    np.savez(npz_file, consecutive_successes=cs_list, max_successes=max_successes, total_time=total_time)

if __name__ == "__main__":
    main()