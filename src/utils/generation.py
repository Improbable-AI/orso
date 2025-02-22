import os
import re
import shutil
import subprocess

from openai import OpenAI
from utils.messages import *
from utils.misc import *

SRC_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{SRC_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"


def sample_llm(model, messages, temperature, n):
    client = OpenAI()

    responses = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature = temperature,
        n=n,
    )

    return [responses.choices[i].message.content for i in range(n)]


def check_run_reward(log_dir, target_name, suffix):

    while True:  # this is to make sure that when running multiple jobs, only tracebacks for the current reward function are taken into consideration
        # Run the code here to check that the implemented reward is bug-free
        reward_check_file = f"{log_dir}/_reward_check_{target_name}.txt"
        with open(reward_check_file, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{SRC_ROOT_DIR}/check_reward.py',
                                        '--env', f'check_{target_name}', '--suffix', f'{suffix}'],
                                        stdout=f, stderr=f)
        process.wait()

        # Read the output and look for traceback
        output = file_to_string(reward_check_file)
        traceback = filter_traceback(output)

        if traceback == "" or log_dir.split("-")[-1] in traceback:
            break

    if traceback != "":
        lines = traceback.split("\n")
        print_log("\n".join([line for line in lines if (not line.startswith("Traceback") and not line.startswith(" ") and not line == "")]))
        
    return traceback


def generate_rewards(log_dir, llm_params, pruned_code, messages, target_task, suffix, attempts=5, n=1, idxes_to_resample=None, evo=-1, timestamp=""):

    assert timestamp != ""

    target_name = re.sub(r'(?<!^)(?=[A-Z])', '_', target_task).lower()

    new_code = pruned_code
    new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")

    successful_samples = 0

    valid_responses = []
    valid_code_strings = []

    if idxes_to_resample is None:
        idxes_to_resample = np.arange(n)

    # Make YAML for train
    original_train_config = file_to_string(f"{ISAAC_ROOT_DIR}/cfg/train/{target_task}PPO.yaml")
    with open(f"{ISAAC_ROOT_DIR}/cfg/train/Check{target_task}{timestamp.split('-')[-1]}{suffix}PPO.yaml", "w") as file:
        new_yaml = original_train_config.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")
        file.write(new_yaml)
    # Make YAML file for task
    original_task_config = file_to_string(f"{ISAAC_ROOT_DIR}/cfg/task/{target_task}.yaml")
    with open(f"{ISAAC_ROOT_DIR}/cfg/task/Check{target_task}{timestamp.split('-')[-1]}{suffix}.yaml", 'w') as file:
        file.write(original_task_config.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}"))

    for a in range(attempts):
        try:
            print_log(f"Attempt {a+1} of {attempts} to generate {n} reward functions.")
            responses = sample_llm(
                model = llm_params["model"],
                messages = messages,
                temperature = llm_params["temperature"],
                n = llm_params["samples"],
            )
            print_log(f"Sampled {len(responses)} responses from LLM.\n--------")

            for i, r in enumerate(responses, 1):
                code_string = parse_python_code(r)

                # Remove unnecessary imports
                lines = code_string.split("\n")
                min_j = 1_000_000
                for j, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        min_j = min(min_j, j)
                        if line.strip().startswith("def compute_reward("):
                            lines[j] = lines[j].replace("def compute_reward(", "def compute_gpt_reward(")
                            code_string = "\n".join(lines[min_j:])

                # The following code is adapted from Eureka's code https://github.com/eureka-research/Eureka/blob/main/eureka/eureka.py
                try:
                    gpt_reward_signatures, input_lsts = get_function_signatures(code_string)
                except:
                    traceback = "Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                    print_log(f"Cannot parse the function signature for sample {i}.\n--------")
                    continue

                for gpt_reward_signature in gpt_reward_signatures:
                    reward_signature = [
                        f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                        f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                        f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                    ]
                    reward_signature = "\n".join([2 * "    " + line for line in reward_signature])
                    if "def compute_reward(self)" in new_code:
                        task_code_string_iter = new_code.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
                    elif "def compute_reward(self, actions)" in new_code:
                        task_code_string_iter = new_code.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
                    else:
                        traceback = "NotImplementedError."
                        print_log(f"NotImplementedError for sample {i}.\n--------")
                        continue

                    # Save the new environment code when the output contains valid code string!
                    with open(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py", 'w') as file:
                        file.writelines(task_code_string_iter + '\n\n')
                        file.writelines("from typing import Tuple, Dict" + '\n')
                        file.writelines("import math" + '\n')
                        file.writelines("import torch" + '\n')
                        file.writelines("from torch import Tensor" + '\n')
                        if "@torch.jit.script" not in code_string:
                            code_string = "@torch.jit.script\n" + code_string
                        file.writelines(code_string + '\n')

                    traceback = check_run_reward(log_dir, target_name, f"{timestamp.split('-')[-1]}{suffix}")
                    if traceback == "":
                        break

                if traceback == "":
                    print_log(f"Sample {i} successful. Successful samples: {successful_samples+1}/{n}\n--------")

                    # Save the full environment code
                    full_code = file_to_string(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
                    full_code = full_code.replace(f"Check{target_task}{timestamp.split('-')[-1]}{suffix}", f"{target_task}{suffix}")

                    with open(f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}_full.py", 'w') as file:
                        file.writelines(full_code)

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}_full.py",
                        f"{log_dir}/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}_full.py",
                    )

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}_full.py",
                        f"{SRC_ROOT_DIR}/envs/{timestamp}/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}_full.py",
                    )

                    # Find compute_reward(self...)
                    pattern = re.compile(f"\s*def compute_reward\(self.*?\):(.+?)(?=\s*def |\Z)", re.DOTALL)
                    match = pattern.search(file_to_string(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"))
                    class_method_code = match.group(0)
                    class_method_code = class_method_code.replace("\n    ", f"\n")
                    if class_method_code.split("\n")[-1].startswith("@torch.jit.script"):
                        class_method_code = "\n".join(class_method_code.split("\n")[:-1])

                    # Save the code
                    with open(f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}.py", 'w') as file:
                        file.writelines("import numpy as np" + "\n")
                        file.writelines("from isaacgym import gymapi, gymtorch" + "\n")
                        file.writelines("from isaacgym.torch_utils import *" + "\n")
                        file.writelines("from isaacgymenvs.utils.torch_jit_utils import *" + "\n")
                        file.writelines("import math" + "\n")
                        file.writelines("import torch" + "\n")
                        file.writelines(f"from isaacgymenvs.tasks.{target_name}{suffix} import *" + "\n")
                        file.writelines(class_method_code + "\n\n")

                        file.writelines("from typing import Tuple, Dict" + '\n')
                        file.writelines("import math" + '\n')
                        file.writelines("import torch" + '\n')
                        file.writelines("from torch import Tensor" + '\n')
                        if "@torch.jit.script" not in code_string:
                            code_string = "@torch.jit.script\n" + code_string
                        file.writelines(code_string + '\n')

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}.py",
                        f"{log_dir}/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}.py",
                    )

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}.py",
                        f"{SRC_ROOT_DIR}/envs/{timestamp}/{target_name}{suffix}_{evo}_{idxes_to_resample[successful_samples]}.py",
                    )

                    valid_responses.append(r)
                    valid_code_strings.append(code_string)
                    successful_samples += 1

                    if successful_samples == n:
                        # Clear the traceback file as it might get too long
                        with open(f"{log_dir}/_prev_tracebacks.txt", "w") as f:
                            f.write("")
                        if os.path.exists(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"):
                            os.remove(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
                        return valid_responses, valid_code_strings
                    new_code = pruned_code
                    new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")
                else:
                    print_log(f"Runtime error in sample {i}.\n--------")
                    with open(f"{log_dir}/_prev_tracebacks.txt", "a") as f:
                        f.write(traceback + "\n")
                    new_code = pruned_code
                    new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")

        except:
            print_log(f"Failed attempt {a+1} of {attempts} to generate a valid reward function (LLM failure). Retrying...")
            new_code = pruned_code
            new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")
            if a == attempts - 1:
                print_log(f"Failed all {attempts} attempts to generate a valid reward function (LLM failure). Exiting...")
                if os.path.exists(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"):
                    os.remove(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
                exit()
            continue

    print_log(f"Failed all {attempts} attempts to generate a valid reward function (LLM failure). Exiting...")
    if os.path.exists(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"):
        os.remove(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
    exit()


def generate_rewards_no_evo(log_dir, llm_params, pruned_code, messages, target_task, suffix, attempts=5, n=1, idxes_to_resample=None, timestamp=""):

    assert timestamp != ""

    target_name = re.sub(r'(?<!^)(?=[A-Z])', '_', target_task).lower()

    new_code = pruned_code
    new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")

    successful_samples = 0

    valid_responses = []
    valid_code_strings = []

    if idxes_to_resample is None:
        idxes_to_resample = np.arange(n)

    # Make YAML for train
    original_train_config = file_to_string(f"{ISAAC_ROOT_DIR}/cfg/train/{target_task}PPO.yaml")
    with open(f"{ISAAC_ROOT_DIR}/cfg/train/Check{target_task}{timestamp.split('-')[-1]}{suffix}PPO.yaml", "w") as file:
        new_yaml = original_train_config.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")
        file.write(new_yaml)
    # Make YAML file for task
    original_task_config = file_to_string(f"{ISAAC_ROOT_DIR}/cfg/task/{target_task}.yaml")
    with open(f"{ISAAC_ROOT_DIR}/cfg/task/Check{target_task}{timestamp.split('-')[-1]}{suffix}.yaml", 'w') as file:
        file.write(original_task_config.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}"))

    for a in range(attempts):
        try:
            print_log(f"Attempt {a+1} of {attempts} to generate {n} reward functions.")
            responses = sample_llm(
                model = llm_params["model"],
                messages = messages,
                temperature = llm_params["temperature"],
                n = llm_params["samples"],
            )
            print_log(f"Sampled {len(responses)} responses from LLM.\n--------")

            for i, r in enumerate(responses, 1):
                code_string = parse_python_code(r)

                # Remove unnecessary imports
                lines = code_string.split("\n")
                min_j = 1_000_000
                for j, line in enumerate(lines):
                    if line.strip().startswith("def "):
                        min_j = min(min_j, j)
                        if line.strip().startswith("def compute_reward("):
                            lines[j] = lines[j].replace("def compute_reward(", "def compute_gpt_reward(")
                            code_string = "\n".join(lines[min_j:])

                # The following code is adapted from Eureka's code https://github.com/eureka-research/Eureka/blob/main/eureka/eureka.py
                try:
                    gpt_reward_signatures, input_lsts = get_function_signatures(code_string)
                except:
                    traceback = "Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                    print_log(f"Cannot parse the function signature for sample {i}.\n--------")
                    continue

                for gpt_reward_signature in gpt_reward_signatures:
                    reward_signature = [
                        f"self.rew_buf[:], self.rew_dict = {gpt_reward_signature}",
                        f"self.extras['gpt_reward'] = self.rew_buf.mean()",
                        f"for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()",
                    ]
                    reward_signature = "\n".join([2 * "    " + line for line in reward_signature])
                    if "def compute_reward(self)" in new_code:
                        task_code_string_iter = new_code.replace("def compute_reward(self):", "def compute_reward(self):\n" + reward_signature)
                    elif "def compute_reward(self, actions)" in new_code:
                        task_code_string_iter = new_code.replace("def compute_reward(self, actions):", "def compute_reward(self, actions):\n" + reward_signature)
                    else:
                        traceback = "NotImplementedError."
                        print_log(f"NotImplementedError for sample {i}.\n--------")
                        continue

                    # Save the new environment code when the output contains valid code string!
                    with open(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py", 'w') as file:
                        file.writelines(task_code_string_iter + '\n\n')
                        file.writelines("from typing import Tuple, Dict" + '\n')
                        file.writelines("import math" + '\n')
                        file.writelines("import torch" + '\n')
                        file.writelines("from torch import Tensor" + '\n')
                        if "@torch.jit.script" not in code_string:
                            code_string = "@torch.jit.script\n" + code_string
                        file.writelines(code_string + '\n')

                    traceback = check_run_reward(log_dir, target_name, f"{timestamp.split('-')[-1]}{suffix}")
                    if traceback == "":
                        break

                if traceback == "":
                    print_log(f"Sample {i} successful. Successful samples: {successful_samples+1}/{n}\n--------")

                    # Save the full environment code
                    full_code = file_to_string(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
                    full_code = full_code.replace(f"Check{target_task}{timestamp.split('-')[-1]}{suffix}", f"{target_task}{suffix}")

                    with open(f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{idxes_to_resample[successful_samples]}_full.py", 'w') as file:
                        file.writelines(full_code)

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{idxes_to_resample[successful_samples]}_full.py",
                        f"{log_dir}/{target_name}{suffix}_{idxes_to_resample[successful_samples]}_full.py",
                    )

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{idxes_to_resample[successful_samples]}_full.py",
                        f"{SRC_ROOT_DIR}/envs/{timestamp}/{target_name}{suffix}_{idxes_to_resample[successful_samples]}_full.py",
                    )

                    # Find compute_reward(self...)
                    pattern = re.compile(f"\s*def compute_reward\(self.*?\):(.+?)(?=\s*def |\Z)", re.DOTALL)
                    match = pattern.search(file_to_string(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"))
                    class_method_code = match.group(0)
                    class_method_code = class_method_code.replace("\n    ", f"\n")
                    if class_method_code.split("\n")[-1].startswith("@torch.jit.script"):
                        class_method_code = "\n".join(class_method_code.split("\n")[:-1])

                    # Save the code
                    with open(f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{idxes_to_resample[successful_samples]}.py", 'w') as file:
                        file.writelines("import numpy as np" + "\n")
                        file.writelines("from isaacgym import gymapi, gymtorch" + "\n")
                        file.writelines("from isaacgym.torch_utils import *" + "\n")
                        file.writelines("from isaacgymenvs.utils.torch_jit_utils import *" + "\n")
                        file.writelines("import math" + "\n")
                        file.writelines("import torch" + "\n")
                        file.writelines(f"from isaacgymenvs.tasks.{target_name}{suffix} import *" + "\n")
                        file.writelines(class_method_code + "\n\n")

                        file.writelines("from typing import Tuple, Dict" + '\n')
                        file.writelines("import math" + '\n')
                        file.writelines("import torch" + '\n')
                        file.writelines("from torch import Tensor" + '\n')
                        if "@torch.jit.script" not in code_string:
                            code_string = "@torch.jit.script\n" + code_string
                        file.writelines(code_string + '\n')

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{idxes_to_resample[successful_samples]}.py",
                        f"{log_dir}/{target_name}{suffix}_{idxes_to_resample[successful_samples]}.py",
                    )

                    shutil.copy(
                        f"{SRC_ROOT_DIR}/envs/{target_name}{suffix}_{idxes_to_resample[successful_samples]}.py",
                        f"{SRC_ROOT_DIR}/envs/{timestamp}/{target_name}{suffix}_{idxes_to_resample[successful_samples]}.py",
                    )

                    valid_responses.append(r)
                    valid_code_strings.append(code_string)
                    successful_samples += 1

                    if successful_samples == n:
                        # Clear the traceback file as it might get too long
                        with open(f"{log_dir}/_prev_tracebacks.txt", "w") as f:
                            f.write("")
                        if os.path.exists(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"):
                            os.remove(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
                        return valid_responses, valid_code_strings
                    new_code = pruned_code
                    new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")
                else:
                    print_log(f"Runtime error in sample {i}.\n--------")
                    lines = traceback.split("\n")
                    lines_to_print = "\n".join([line for line in lines if (not line.startswith("Traceback") and not line.startswith(" ") and not line == "")])
                    if lines_to_print not in file_to_string(f"{log_dir}/_prev_tracebacks.txt"):
                        with open(f"{log_dir}/_prev_tracebacks.txt", "a") as f:
                            f.write(traceback + "\n")
                    new_code = pruned_code
                    new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")

        except:
            print_log(f"Failed attempt {a+1} of {attempts} to generate a valid reward function (LLM failure). Retrying...")
            new_code = pruned_code
            new_code = new_code.replace(target_task, f"Check{target_task}{timestamp.split('-')[-1]}{suffix}")
            if a == attempts - 1:
                print_log(f"Failed all {attempts} attempts to generate a valid reward function (LLM failure). Exiting...")
                if os.path.exists(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"):
                    os.remove(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
                exit()
            continue

    print_log(f"Failed all {attempts} attempts to generate a valid reward function (LLM failure). Exiting...")
    if os.path.exists(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py"):
        os.remove(f"{ISAAC_ROOT_DIR}/tasks/_check_{target_name}{timestamp.split('-')[-1]}{suffix}.py")
    exit()
