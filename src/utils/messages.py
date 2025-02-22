from utils.misc import *

SRC_ROOT_DIR = os.getcwd()
ISAAC_ROOT_DIR = f"{SRC_ROOT_DIR}/../isaacgymenvs/isaacgymenvs"

def init_messages(obs_code, description):
    prompt_dir = f"{SRC_ROOT_DIR}/prompts"

    init_system = file_to_string(f"{prompt_dir}/init_system.txt")
    init_user = file_to_string(f"{prompt_dir}/init_user.txt")
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    reward_signature = file_to_string(f'{prompt_dir}/reward_signature.txt')

    init_system = init_system.format(task_reward_signature_string=reward_signature)
    init_system += code_output_tip
    init_user = init_user.format(task_obs_code_string=obs_code, task_description=description)

    messages = [
        {"role": "system", "content": init_system},
        {"role": "user", "content": init_user},
    ]

    return messages


def update_messages(log_dir, messages, response, stats, epoch_freq):
    prompt_dir = f"{SRC_ROOT_DIR}/prompts"

    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')

    content = ""
    content += policy_feedback.format(epoch_freq=epoch_freq)
    content += ("\n" + stats + "\n")
    content += code_feedback + "\n"
    content += code_output_tip

    if len(messages) < 4:
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": content})
    else:
        assert len(messages) == 4, f"Messages length should be 4, but it's {len(messages)}"
        messages[-2] = {"role": "assistant", "content": response}
        messages[-1] = {"role": "user", "content": content}

    return messages