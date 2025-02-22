import ast
import contextlib
import logging
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from tqdm import tqdm


def print_log(s):
    """Prints a string to the console and logs it to the log file."""
    print(s)
    logging.info(s)


def file_to_string(filename):
    """Reads a file and returns its contents as a string."""
    with open(filename, "r", encoding='utf-8') as f:
        return f.read()


def filter_traceback(s):
    """Filters the traceback from a string."""
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if "Traceback" in line:
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


def parse_python_code(s):
    """Parses python code from a string."""
    patterns = [
        r'```python(.*?)```',
        r'```(.*?)```',
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    for pattern in patterns:
        code_string = re.search(pattern, s, re.DOTALL)
        if code_string is not None:
            code_string = code_string.group(1).strip()
            break
    code_string = s if not code_string else code_string

    return code_string


def get_function_signature(code_string):
    """Gets the function signature from a code string."""
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    # For simplicity, we'll just return the signature of the first function definition
    function_def = function_defs[0]

    input_lst = []
    # Construct the function signature (within object class)
    signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
    for arg in function_def.args.args:
        input_lst.append(arg.arg)
    return signature, input_lst


def get_function_signatures(code_string):
    """Gets the function signature from a code string."""
    # Parse the code string into an AST
    module = ast.parse(code_string)

    # Find the function definitions
    function_defs = [node for node in module.body if isinstance(node, ast.FunctionDef)]

    # If there are no function definitions, return None
    if not function_defs:
        return None

    input_lsts = []
    signatures = []
    for function_def in function_defs:
        input_lst = []
        # Construct the function signature (within object class)
        signature = function_def.name + '(self.' + ', self.'.join(arg.arg for arg in function_def.args.args) + ')'
        signatures.append(signature)
        for arg in function_def.args.args:
            input_lst.append(arg.arg)
        input_lsts.append(input_lst)
    return signatures, input_lsts


def block_until_training(rl_filepath, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    while True:
        rl_log = file_to_string(rl_filepath)
        if "global_step: " in rl_log or "Traceback" in rl_log:
            if "global_step: " in rl_log:
                print_log(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if "Traceback" in rl_log:
                print_log(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
    time.sleep(30)


def replace_method(module_name, class_name, method_name, new_method):
    # Import the module dynamically
    module = __import__(module_name, fromlist=[class_name])

    # Get the class dynamically
    class_instance = getattr(module, class_name)

    # Check if the method exists before deleting
    if hasattr(class_instance, method_name):
        # Delete the existing method
        delattr(class_instance, method_name)
        print(f"{module_name}.{class_name}.{method_name} deleted.")

        # Set the new method
        setattr(class_instance, method_name, new_method)
        print(f"{module_name}.{class_name}.{method_name} replaced with the new method.")
    else:
        print(f"{module_name}.{class_name}.{method_name} not found.")


def load_tensorboard_logs(path):
    """Loads tensorboard logs from a directory."""
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data


def set_freest_gpu():
    freest_gpus = find_freest_gpu()
    print_log(f"Freest GPUs: {freest_gpus}")
    ordered_gpu_ids = [str(gpu[0]) for gpu in freest_gpus]
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(ordered_gpu_ids)
    return ordered_gpu_ids


def find_freest_gpu():
    try:
        # Run nvidia-smi command to get GPU information
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        # Check if the command was successful
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split('\n')
            # Parse GPU information
            gpu_memory = [tuple(map(int, line.split(', '))) for line in gpu_info]
            # Sort GPUs based on free memory
            sorted_gpus = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
            # Return the index of the GPU with the most free memory
            return sorted_gpus
        else:
            print("Error running nvidia-smi command.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None