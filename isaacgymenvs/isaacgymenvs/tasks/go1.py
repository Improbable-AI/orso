# Copyright (c) 2018-2023, NVIDIA Corporation
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

import os
from typing import Dict, Tuple

import numpy as np
import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask


class Go1(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 1027
        self.camera_props.height = 768

        self.cfg = cfg
        
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["action_smoothness"] = self.cfg["env"]["learn"]["actionSmoothnessRewardScale"]
        self.rew_scales["joint_vel"] = self.cfg["env"]["learn"]["jointVelRewardScale"]
        self.rew_scales["cosmetic"] = self.cfg["env"]["learn"]["cosmeticRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["power"] = self.cfg["env"]["learn"]["powerRewardScale"]
        self.rew_scales["foot_clearance"] = self.cfg["env"]["learn"]["footClearanceRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["base_orientation"] = self.cfg["env"]["learn"]["baseOrientationRewardScale"]
        self.rew_scales["tracking_contacts_force"] = self.cfg["env"]["learn"]["trackingContactsForceRewardScale"]
        self.rew_scales["tracking_contacts_vel"] = self.cfg["env"]["learn"]["trackingContactsVelRewardScale"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 42
        self.cfg["env"]["numActions"] = 12

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:self.num_envs, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)[:self.num_envs * self.num_bodies, :]

        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        
        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # initialize some data used later on
        self.extras = {}
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)


    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "urdf/go1/urdf/go1.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = 3
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        go1_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(go1_asset)
        self.num_actuated_dof = self.num_actions
        self.num_bodies = self.gym.get_asset_rigid_body_count(go1_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(go1_asset)
        # rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(go1_asset)

        body_names = self.gym.get_asset_rigid_body_names(go1_asset)
        self.dof_names = self.gym.get_asset_dof_names(go1_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)

        feet_names = [s for s in body_names if "foot" in s]
        penalized_contact_names = []
        for name in ["thigh", "calf"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in ["base", "trunk", "hip"]:
            termination_contact_names.extend([s for s in body_names if name in s])

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.go1_handles = []
        self.envs = []

        for i in range(self.num_dof):
            dof_props_asset['driveMode'][i] = gymapi.DOF_MODE_POS
            dof_props_asset['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            dof_props_asset['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd

        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        for s in range(len(dof_props_asset)):
            self.dof_pos_limits[s, 0] = dof_props_asset["lower"][s].item()
            self.dof_pos_limits[s, 1] = dof_props_asset["upper"][s].item()
            self.dof_vel_limits[s] = dof_props_asset["velocity"][s].item()
            self.torque_limits[s] = dof_props_asset["effort"][s].item()
            # soft limits
            m = (self.dof_pos_limits[s, 0] + self.dof_pos_limits[s, 1]) / 2
            r = self.dof_pos_limits[s, 1] - self.dof_pos_limits[s, 0]
            self.dof_pos_limits[s, 0] = m - 0.5 * r * 0.9
            self.dof_pos_limits[s, 1] = m + 0.5 * r * 0.9

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            go1_handle = self.gym.create_actor(env_ptr, go1_asset, start_pose, "go1", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, go1_handle, dof_props_asset)
            self.gym.enable_actor_dof_force_sensors(env_ptr, go1_handle)
            self.envs.append(env_ptr)
            self.go1_handles.append(go1_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go1_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go1_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.go1_handles[0], termination_contact_names[i])

        # necessary to save videos
        self.rendering_camera = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(1, 1, 1), gymapi.Vec3(0, 0, 0))
        self.video_frames = []

    def render(self, mode="rgb_array"):
        assert mode == "rgb_array"
        bx, by, bz = self.root_states[0, 0], self.root_states[0, 1], self.root_states[0, 2]
        self.gym.set_camera_location(self.rendering_camera, self.envs[0], gymapi.Vec3(bx+1, by+1, bz+1), gymapi.Vec3(bx, by, bz))
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        img = self.gym.get_camera_image(self.sim, self.envs[0], self.rendering_camera, gymapi.IMAGE_COLOR)
        w, h = img.shape
        img_reshaped = img.reshape([w, h // 4, 4])
        # Permute BRGA to RGBA
        img_permuted = img_reshaped[:, :, [2, 1, 0, 3]]
        return img_permuted

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = torch.clamp(self.action_scale * self.actions + self.default_dof_pos, self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1])
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.consecutive_successes[:] = compute_anymal_reward(
            # tensors
            self.root_states,
            self.commands,
            self.torques,
            self.contact_forces,
            self.foot_velocities,
            self.foot_positions,
            self.dof_pos,
            self.dof_pos_limits,
            self.penalised_contact_indices,
            self.termination_contact_indices,
            self.gravity_vec,
            self.consecutive_successes,
            self.progress_buf,
            # Dict
            self.rew_scales,
            # other
            self.max_episode_length,
        )

        self.extras['consecutive_successes'] = self.consecutive_successes.mean() 

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.obs_buf[:] = compute_go1_observations( # tensors
                                                    self.root_states,
                                                    self.commands,
                                                    self.dof_pos,
                                                    self.default_dof_pos,
                                                    self.dof_vel,
                                                    self.gravity_vec,
                                                    self.actions,
                                                    # scales
                                                    self.lin_vel_scale,
                                                    self.ang_vel_scale,
                                                    self.dof_pos_scale,
                                                    self.dof_vel_scale
        )

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # do not modify from here ...
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # ... to here

        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()

        # do not modify from here ...
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # ... to here

@torch.jit.script
def compute_go1_observations(root_states,
        commands,
        dof_pos,
        default_dof_pos,
        dof_vel,
        gravity_vec,
        actions,
        lin_vel_scale,
        ang_vel_scale,
        dof_pos_scale,
        dof_vel_scale
    ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    # base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) * lin_vel_scale
    # base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13]) * ang_vel_scale
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    dof_pos_scaled = (dof_pos - default_dof_pos) * dof_pos_scale

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)

    obs = torch.cat((projected_gravity,
                     commands_scaled,
                     dof_pos_scaled,
                     dof_vel*dof_vel_scale,
                     actions,
                    #  base_lin_vel,
                    #  base_ang_vel,
                     ), dim=-1)

    return obs

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_anymal_reward(
    # tensors
    root_states,
    commands,
    torques,
    contact_forces,
    foot_velocities,
    foot_positions,
    dof_pos,
    dof_pos_limits,
    penalised_contact_indices,
    termination_contact_indices,
    gravity_vec,
    consecutive_successes,
    episode_lengths,
    # Dict
    rew_scales,
    # other
    max_episode_length
):
    # (reward, reset, feet_in air, feet_air_time, episode sums)
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int) -> Tuple[Tensor, Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]

    rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * rew_scales["lin_vel_z"]
    rew_ang_vel_xy = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1) * rew_scales["ang_vel_xy"]

    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1) * rew_scales["torque"]

    foot_heights = foot_positions[:, :, 2]
    foot_vel_xys = torch.norm(foot_velocities[:, :, 0:2], dim=-1)
    foot_target_height = 0.1 # 0.08

    rew_foot_clearance = (
        torch.sum((foot_heights - foot_target_height)**2 * foot_vel_xys, dim=-1) * rew_scales["foot_clearance"]
    )

    base_heights = root_states[:, 2]
    rew_base_height = torch.square(base_heights - 0.34) * rew_scales["base_height"]

    projected_gravity = quat_rotate(base_quat, gravity_vec)
    rew_ori = torch.sum(torch.square(projected_gravity[:, 0:2]), dim=-1) * rew_scales["base_orientation"]

    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_base_height + rew_base_height + rew_torque + rew_lin_vel_z + rew_ang_vel_xy + rew_ori + rew_foot_clearance
    # total_reward = base_lin_vel[:, 0] + 2 * (rew_ang_vel_z + rew_base_height + rew_base_height + rew_torque + rew_lin_vel_z + rew_ang_vel_xy + rew_ori + rew_foot_clearance)
    total_reward += 0.05 # survival bonus
    total_reward = torch.clip(total_reward, 0.0, None)

    dof_lower_limits_reset = torch.any(dof_pos < dof_pos_limits[:, 0], dim=1)
    dof_upper_limits_reset = torch.any(dof_pos > dof_pos_limits[:, 1], dim=1)
    dof_limits_reset = torch.logical_or(dof_upper_limits_reset, dof_lower_limits_reset)

    # height_reset = torch.where(base_heights < 0.3, torch.ones_like(base_heights), torch.zeros_like(base_heights))

    # reset agents
    reset = torch.any(torch.norm(contact_forces[:, termination_contact_indices, :], dim=-1) > 1., dim=1)
    reset = reset | torch.any(torch.norm(contact_forces[:, penalised_contact_indices, :], dim=-1) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    reset = reset | time_out
    reset = torch.logical_or(reset, dof_limits_reset)
    # reset = torch.logical_or(reset, height_reset)
    
    consecutive_successes = -(lin_vel_error + ang_vel_error).mean()
    # consecutive_successes = base_lin_vel[:, 0].mean()

    # total_reward = -(lin_vel_error + ang_vel_error)
    return total_reward.detach(), reset, consecutive_successes