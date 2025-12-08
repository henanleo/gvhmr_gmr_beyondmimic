"""
合并 demo.py 和 gvhmr_to_robot.py 的功能
从视频输入直接输出机器人运动 csv 文件
"""
import argparse
import pathlib
import os
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# demo.py 的导入
import cv2
import pytorch_lightning as pl
from hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pytorch3d.transforms import quaternion_to_matrix

from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
from scipy.spatial.transform import Rotation as sRot

# gvhmr_to_robot.py 的导入
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_gvhmr_pred_file, get_gvhmr_data_offline_fast

from rich import print

CRF = 23  # 17 is lossless, every +6 halves the mp4 size


def parse_args_to_cfg(video_path, static_cam=False, use_dpvo=False, f_mm=None, verbose=False, output_root=None):
    """从 demo.py 提取的配置解析函数"""
    video_path = Path(video_path)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={static_cam}",
            f"verbose={verbose}",
            f"use_dpvo={use_dpvo}",
        ]
        if f_mm is not None:
            overrides.append(f"f_mm={f_mm}")

        # Allow to change output root
        if output_root is not None:
            overrides.append(f"output_root={output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    """从 demo.py 提取的预处理函数"""
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:  # DPVO
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time() - tic:.2f}s")


def load_data_dict(cfg):
    """从 demo.py 提取的数据加载函数"""
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data


def run_hmr4d(cfg):
    """运行 HMR4D 预测，返回结果文件路径"""
    paths = cfg.paths
    
    # ===== Preprocess and save to disk ===== #
    run_preprocess(cfg)
    data = load_data_dict(cfg)

    # ===== HMR4D ===== #
    if not Path(paths.hmr4d_results).exists():
        Log.info("[HMR4D] Predicting")
        model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        tic = Log.sync_time()
        pred = model.predict(data, static_cam=cfg.static_cam)
        pred = detach_to_cpu(pred)
        data_time = data["length"] / 30
        Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
        torch.save(pred, paths.hmr4d_results)
    else:
        Log.info(f"[HMR4D] Results already exist at {paths.hmr4d_results}")
    
    return paths.hmr4d_results


def video_to_robot_csv(video_path, output_csv_path, robot="unitree_g1", 
                       static_cam=False, use_dpvo=False, f_mm=None, 
                       verbose=False, output_root=None, visualize=False,
                       record_video=False, rate_limit=False):
    """
    主函数：从视频输入到机器人 csv 文件
    
    参数:
        video_path: 输入视频路径
        output_csv_path: 输出 csv 文件路径
        robot: 机器人类型
        static_cam: 是否使用静态相机
        use_dpvo: 是否使用 DPVO
        f_mm: 焦距（毫米）
        verbose: 是否显示详细信息
        output_root: 输出根目录
        visualize: 是否可视化（显示机器人运动）
        record_video: 是否录制视频
        rate_limit: 是否限制播放速率
    """
    HERE = pathlib.Path(__file__).parent
    
    # 步骤 1: 运行 HMR4D 从视频生成预测结果
    Log.info("="*60)
    Log.info("步骤 1: 运行 HMR4D 预测")
    Log.info("="*60)
    cfg = parse_args_to_cfg(video_path, static_cam, use_dpvo, f_mm, verbose, output_root)
    hmr4d_results_path = run_hmr4d(cfg)
    
    # 步骤 2: 从 HMR4D 结果生成机器人运动并保存为 csv
    Log.info("="*60)
    Log.info("步骤 2: 运动重定向到机器人并保存为 csv")
    Log.info("="*60)
    
    SMPLX_FOLDER = HERE / "GMR" / "assets" / "body_models"
    
    # Load SMPLX trajectory
    smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(
        hmr4d_results_path, SMPLX_FOLDER
    )
    
    # align fps
    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(smplx_data, body_model, smplx_output, tgt_fps=tgt_fps)
    
    # Initialize the retargeting system
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=robot,
    )
    
    # 准备保存目录
    if output_csv_path is None:
        base_name = os.path.splitext(video_path)[0]
        output_csv_path = base_name + ".csv"
    else:
        _, ext = os.path.splitext(output_csv_path)
        if ext == "":
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_csv_path = os.path.join(output_csv_path, video_name + ".csv")

    # ---- 确保目录存在 ----
    save_dir = os.path.dirname(output_csv_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    qpos_list = []
    
    # 如果需要可视化，初始化 viewer
    robot_motion_viewer = None
    if visualize:
        video_name = Path(video_path).stem
        robot_motion_viewer = RobotMotionViewer(
            robot_type=robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=record_video,
            video_path=f"videos/{robot}_{video_name}.mp4",
        )
    
    # 处理每一帧
    Log.info(f"处理 {len(smplx_data_frames)} 帧...")
    for i in tqdm(range(len(smplx_data_frames)), desc="重定向运动"):
        smplx_data = smplx_data_frames[i]
        
        # retarget
        qpos = retarget.retarget(smplx_data)
        
        # 如果需要可视化
        if visualize and robot_motion_viewer is not None:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=rate_limit,
            )
        
        # 保存 qpos
        qpos_list.append(qpos)
    
    # 保存为 csv 文件
    root_pos = np.array([qpos[:3] for qpos in qpos_list])
    # save from wxyz to xyzw
    root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
    dof_pos = np.array([qpos[7:] for qpos in qpos_list])
    local_body_pos = None
    body_names = None

    combined_data = np.hstack([root_pos, root_rot, dof_pos])

    import csv
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入数据
            for row in combined_data:
                writer.writerow(row)
    print(f"✓ 已保存到 {output_csv_path}")
    
    # 关闭 viewer
    if visualize and robot_motion_viewer is not None:
        robot_motion_viewer.close()
    
    Log.info("="*60)
    Log.info("完成！")
    Log.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="从视频输入直接生成机器人运动 csv 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 视频输入参数
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输入视频路径"
    )
    
    # 输出参数
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出 csv 文件路径"
    )
    
    # 机器人参数
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
                 "booster_t1", "booster_t1_29dof","stanford_toddy", "fourier_n1", 
                "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro", "berkeley_humanoid_lite", "booster_k1",
                "pnd_adam_lite", "openloong", "tienkung"],
        default="unitree_g1",
        help="机器人类型"
    )
    
    # HMR4D 相关参数
    parser.add_argument(
        "-s", "--static_cam",
        action="store_true",
        help="如果为真，跳过 DPVO（使用静态相机）"
    )
    parser.add_argument(
        "--use_dpvo",
        action="store_true",
        help="如果为真，使用 DPVO。默认不使用 DPVO。"
    )
    parser.add_argument(
        "--f_mm",
        type=int,
        default=None,
        help="全画幅相机焦距（毫米）。留空使用默认值。"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="如果为真，绘制中间结果"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="输出根目录，默认为 outputs/demo"
    )
    
    # 可视化参数
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="如果为真，可视化机器人运动（显示窗口）"
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="如果为真，录制机器人运动视频（需要 --visualize）"
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="如果为真，限制播放速率以保持与人类运动相同的速度（需要 --visualize）"
    )
    
    args = parser.parse_args()
    
    # 运行主函数
    video_to_robot_csv(
        video_path=args.video,
        output_csv_path=args.output_csv,
        robot=args.robot,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
        verbose=args.verbose,
        output_root=args.output_root,
        visualize=args.visualize,
        record_video=args.record_video,
        rate_limit=args.rate_limit,
    )

