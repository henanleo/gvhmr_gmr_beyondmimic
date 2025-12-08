"""
useage:python preprocessing.py \
    --amass_file AMASS/BMLrub_walk.npz \
    --output_file motions/walk_motion.npz \
    --input_fps 60 \
    --output_fps 50
"""
import numpy as np
import torch

from isaaclab.utils.math import axis_angle_from_quat, quat_from_axis_angle, quat_conjugate, quat_mul, quat_slerp

class AMASS2NPZ:
    def __init__(self, amass_file, output_file, input_fps=60, output_fps=50, device='cpu', frame_range=None):
        self.amass_file = amass_file
        self.output_file = output_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.device = torch.device(device)
        self.frame_range = frame_range
        self._load_amass()
        self._interpolate_motion()
        self._compute_velocities()
        self._save_npz()

    def _load_amass(self):
        data = np.load(self.amass_file)
        poses = data['poses']         # (T, 72)
        trans = data['trans']         # (T, 3)
        
        if self.frame_range is not None:
            start, end = self.frame_range
            poses = poses[start:end+1]
            trans = trans[start:end+1]

        poses = torch.tensor(poses, dtype=torch.float32, device=self.device)
        trans = torch.tensor(trans, dtype=torch.float32, device=self.device)

        # 根节点旋转 (axis-angle) -> quaternion
        root_rot_aa = poses[:, :3]  # 根关节
        root_quat = quat_from_axis_angle(root_rot_aa)  # shape (T,4)

        # 关节 DOF
        dof_pos = poses[:, 3:]  # shape (T, 69)

        self.input_frames = poses.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

        self.motion_base_poss_input = trans
        self.motion_base_rots_input = root_quat
        self.motion_dof_poss_input = dof_pos

        print(f"[INFO] Loaded AMASS motion: {self.amass_file}, frames: {self.input_frames}, duration: {self.duration:.2f}s")

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device)
        self.output_frames = times.shape[0]

        index_0, index_1, blend = self._compute_frame_blend(times)

        self.motion_base_poss = self._lerp(self.motion_base_poss_input[index_0],
                                           self.motion_base_poss_input[index_1],
                                           blend.unsqueeze(1))
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[index_0],
                                            self.motion_base_rots_input[index_1],
                                            blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[index_0],
                                          self.motion_dof_poss_input[index_1],
                                          blend.unsqueeze(1))
        print(f"[INFO] Interpolated to {self.output_frames} frames at {self.output_fps} FPS")

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _save_npz(self):
        log = {
            "fps": np.array([self.output_fps]),
            "joint_pos": self.motion_dof_poss.cpu().numpy(),
            "joint_vel": self.motion_dof_vels.cpu().numpy(),
            "body_pos_w": self.motion_base_poss.cpu().numpy(),
            "body_quat_w": self.motion_base_rots.cpu().numpy(),
            "body_lin_vel_w": self.motion_base_lin_vels.cpu().numpy(),
            "body_ang_vel_w": self.motion_base_ang_vels.cpu().numpy()
        }
        np.savez(self.output_file, **log)
        print(f"[INFO] Saved motion npz to: {self.output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--input_fps", type=int, default=60)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--frame_range", type=int, nargs=2)
    args = parser.parse_args()

    converter = AMASS2NPZ(args.amass_file, args.output_file, args.input_fps,
                          args.output_fps, args.device, args.frame_range)
