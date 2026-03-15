import argparse
import copy
import datetime
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

import storm.models as models
import storm.utils.misc as misc
import imageio
from PIL import Image

from storm.dataset.constants import DATASET_DICT, MEAN, STD
from storm.dataset.data_utils import prepare_inputs_and_targets, to_batch_tensor
from storm.dataset.storm_dataset import SingleSequenceDataset
from storm.utils.logging import setup_logging
from storm.visualization.video_maker import make_video


def extrapolate_camtoworlds(camtoworlds, n_extra):
    """Extrapolate future camera poses using constant-velocity model (translation only).

    camtoworlds: [B, T, V, 4, 4]  (FLU canonical convention)
    Returns:     [B, T + n_extra, V, 4, 4]
    """
    # Velocity = displacement of last step
    vel = camtoworlds[:, -1:, :, :3, 3] - camtoworlds[:, -2:-1, :, :3, 3]  # [B, 1, V, 3]
    extra_list = []
    for i in range(1, n_extra + 1):
        new_pose = camtoworlds[:, -1:].clone()  # [B, 1, V, 4, 4]
        new_pose[:, :, :, :3, 3] = camtoworlds[:, -1:, :, :3, 3] + i * vel
        extra_list.append(new_pose)
    extra = torch.cat(extra_list, dim=1)  # [B, n_extra, V, 4, 4]
    return torch.cat([camtoworlds, extra], dim=1)  # [B, T + n_extra, V, 4, 4]


def elevate_camtoworlds(camtoworlds, height=2.0, tilt_deg=15.0):
    """Lift cameras up by `height` metres and tilt nose down by `tilt_deg` degrees.

    In STORM's FLU canonical frame (Waymo): X=forward, Y=left, Z=up.
    - Elevation: add `height` to the Z translation component.
    - Tilt down: right-multiply by Ry(tilt_deg) in the local camera frame,
      which rotates the forward (X) axis toward -Z (downward).

    camtoworlds: [..., 4, 4]
    Returns: same shape
    """
    theta = math.radians(tilt_deg)
    c, s = math.cos(theta), math.sin(theta)
    Ry = torch.tensor(
        [[c, 0.0, s, 0.0],
         [0.0, 1.0, 0.0, 0.0],
         [-s, 0.0, c, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=camtoworlds.dtype, device=camtoworlds.device,
    )  # [4, 4]
    elevated = camtoworlds.clone()
    elevated[..., 2, 3] += height   # translate up in world Z (FLU up)
    elevated = elevated @ Ry         # tilt nose down in local frame
    return elevated


def _clone_render_dict(data_dict):
    out = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _build_fullres_render_dict(input_dict, scene_json, num_max_cams, device):
    camera_list = DATASET_DICT[scene_json["dataset"]]["camera_list"][num_max_cams]
    sizes = [tuple(scene_json["original_image_size"][cam]) for cam in camera_list]
    full_h, full_w = sizes[0]
    if any(size != (full_h, full_w) for size in sizes):
        raise ValueError(f"Selected cameras do not share one full resolution: {sizes}")

    full_target_intrinsics = torch.zeros_like(input_dict["target_intrinsics"])
    for v, cam in enumerate(camera_list):
        fx_n, fy_n, cx_n, cy_n = scene_json["normalized_intrinsics"][cam]
        K = torch.tensor(
            [[fx_n * full_w, 0.0, cx_n * full_w],
             [0.0, fy_n * full_h, cy_n * full_h],
             [0.0, 0.0, 1.0]],
            dtype=input_dict["target_intrinsics"].dtype,
            device=device,
        )
        full_target_intrinsics[:, :, v] = K

    full_render_dict = _clone_render_dict(input_dict)
    full_render_dict["target_intrinsics"] = full_target_intrinsics
    full_render_dict["width"] = full_w
    full_render_dict["height"] = full_h
    return full_render_dict, camera_list, full_h, full_w


def _load_fullres_gt_frames(scene_json, data_root, target_frame_indices, camera_list, full_h, full_w):
    gt_frames = []
    num_timesteps = scene_json["num_timesteps"]
    for frame_idx in target_frame_indices:
        frame_views = []
        if 0 <= frame_idx < num_timesteps:
            for cam in camera_list:
                img_relative_path = scene_json["relative_image_path"][cam][frame_idx]
                img_path = os.path.join(data_root, img_relative_path)
                frame_views.append(np.array(Image.open(img_path).convert("RGB")))
        else:
            for _ in camera_list:
                frame_views.append(np.zeros((full_h, full_w, 3), dtype=np.uint8))
        gt_frames.append(frame_views)
    return gt_frames


def _denorm_rendered_images(x, device):
    mean = torch.tensor(MEAN, device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor(STD, device=device).view(1, 1, 1, 1, 3)
    x = (x.float() * std + mean).clamp(0.0, 1.0)
    return x.permute(0, 1, 4, 2, 3)


def _chw_to_uint8(x):
    return (x.permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)


def render_chunked(model, gs_params, input_dict, chunk_size=20):
    """Run model.from_gs_params_to_output in temporal chunks to avoid OOM.

    Splits input_dict["target_camtoworlds"] / intrinsics / time along the T
    dimension, renders each chunk, then concatenates along T.

    Returns a pred_dict whose render_results tensors span all T frames.
    """
    T = input_dict["target_camtoworlds"].shape[1]
    if T <= chunk_size:
        return model.from_gs_params_to_output(gs_params, input_dict)

    chunk_preds = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_dict = dict(input_dict)
        chunk_dict["target_camtoworlds"] = input_dict["target_camtoworlds"][:, start:end]
        chunk_dict["target_intrinsics"] = input_dict["target_intrinsics"][:, start:end]
        if "target_time" in input_dict:
            chunk_dict["target_time"] = input_dict["target_time"][:, start:end]
        chunk_preds.append(model.from_gs_params_to_output(gs_params, chunk_dict))
        torch.cuda.empty_cache()

    # Concatenate render_results tensors along the T dimension (dim=1)
    merged_render = {}
    rr0 = chunk_preds[0]["render_results"]
    for key, val in rr0.items():
        if isinstance(val, torch.Tensor) and val.dim() >= 2:
            merged_render[key] = torch.cat(
                [p["render_results"][key] for p in chunk_preds], dim=1
            )
        else:
            merged_render[key] = val  # string keys like rgb_key, depth_key, etc.
    return {"render_results": merged_render}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def get_args_parser():
    parser = argparse.ArgumentParser("STORM training", add_help=False)

    # =============== Model parameters ================= #
    parser.add_argument("--model", default="STORM-B/8", type=str)
    parser.add_argument("--num_context_timesteps", default=4, type=int)
    parser.add_argument("--num_target_timesteps", default=4, type=int)
    parser.add_argument("--gs_dim", default=3, type=int, help="Number of gs dimensions")
    parser.add_argument("--use_sky_token", action="store_true")
    parser.add_argument("--use_affine_token", action="store_true")
    parser.add_argument("--use_latest_gsplat", action="store_true")
    parser.add_argument(
        "--decoder_type",
        type=str,
        choices=["dummy", "conv"],
        default="dummy",
        help="STORM or LatentSTORM",
    )
    parser.add_argument("--num_motion_tokens", default=16, type=int, help="Number of motion tokens")

    # ============= Checkpoint parameters ============= #
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None, help="resume from checkpoint")
    parser.add_argument("--load_from", type=str, default=None)

    # ============= Dataset parameters ============= #
    parser.add_argument("--data_root", default="./data/STORM2", type=str, help="dataset path")
    parser.add_argument("--overwrite_train_ctx_view_with", default=None, type=int)
    parser.add_argument("--overwrite_train_tgt_view_with", default=None, type=int)
    parser.add_argument("--overwrite_test_ctx_view_with", default=None, type=int)
    parser.add_argument("--input_size", default=(160, 240), type=int, nargs=2)
    parser.add_argument("--num_max_cameras", type=int, default=3)
    parser.add_argument("--timespan", type=float, default=2.0)
    parser.add_argument("--load_ground", action="store_true")
    parser.add_argument("--load_depth", action="store_true")
    parser.add_argument("--load_flow", action="store_true")
    parser.add_argument("--dataset", default="waymo", type=str, choices=DATASET_DICT.keys())
    parser.add_argument("--skip_sky_mask", action="store_true", help="skip sky mask loading")
    parser.add_argument("--scene_list_file", type=str, default=None)
    # ============= Logging ============= #
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--num_vis_samples", type=int, default=1)

    # ============= Rendering ============= #
    parser.add_argument(
        "--num_extra_frames", default=20, type=int,
        help="Number of future frames to extrapolate beyond the sequence end (0 to disable)",
    )
    parser.add_argument(
        "--render_elevated", action="store_true",
        help="Also render a second video with camera elevated and tilted down",
    )
    parser.add_argument("--elevated_height", default=2.0, type=float, help="Elevation in metres")
    parser.add_argument(
        "--elevated_tilt_deg", default=15.0, type=float,
        help="Nose-down tilt in degrees for the elevated view",
    )

    # ============= Miscellaneous ============= #
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")

    # ============= WandB ============= #
    parser.add_argument("--project", default="debug", type=str)
    parser.add_argument("--exp_name", default=None, type=str)

    return parser


def main(args):
    global logger
    args.exp_name = args.model.replace("/", "-") if args.exp_name is None else args.exp_name
    log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    video_dir = os.path.join(log_dir, "videos")
    args.log_dir, args.ckpt_dir, args.video_dir = log_dir, checkpoint_dir, video_dir
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device(args.device)
    seed = args.seed
    misc.fix_random_seeds(seed)
    cudnn.benchmark = True

    # set up logging
    setup_logging(output=log_dir, level=logging.INFO)
    logger = logging.getLogger("STORM")
    logger.info(f"hostname: {os.uname().nodename}\n")
    logger.info(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    logger.info(f"Logging to {log_dir}")
    logger.info(json.dumps(args.__dict__, indent=4, sort_keys=True))
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    dataset_meta = DATASET_DICT[args.dataset]
    train_annotation = args.scene_list_file

    num_context_timesteps = dataset_meta["num_context_timesteps"]
    num_target_timesteps = dataset_meta["num_target_timesteps"]
    if args.overwrite_train_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_train_ctx_view_with
    if args.overwrite_test_ctx_view_with is not None:
        num_context_timesteps = args.overwrite_test_ctx_view_with
    if args.overwrite_train_tgt_view_with is not None:
        num_target_timesteps = args.overwrite_train_tgt_view_with
    input_size = dataset_meta["size"]
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"annotation_txt_file_list_train: {train_annotation}")

    if args.model in models.STORM_models:
        model = models.STORM_models[args.model](
            img_size=args.input_size,
            gs_dim=args.gs_dim,
            decoder_type=args.decoder_type,
            use_sky_token=args.use_sky_token,
            use_affine_token=args.use_affine_token,
            num_motion_tokens=args.num_motion_tokens,
            use_latest_gsplat=args.use_latest_gsplat,
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}")

    logger.info(f"Model = {str(model)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Parameters: {n_params / 1e6:.2f}M ({n_params:,})")
    model.to(device)
    dataset = SingleSequenceDataset(
        data_root=args.data_root,
        annotation_txt_file_list=train_annotation,
        target_size=input_size,
        num_context_timesteps=num_context_timesteps,
        num_target_timesteps=num_target_timesteps,
        timespan=args.timespan,
        num_max_cams=args.num_max_cameras,
        load_depth=args.load_depth,
        load_flow=args.load_flow,
    )

    logger.info(f"Dataset contains {len(dataset):,} sequences using {train_annotation}.")
    misc.load_model(args, model)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"{args.model} Trainable Parameters: {num_trainable_params / 1e6:.2f}M")
    model.eval().cuda()
    logger.info(f"Preparing data... (This may take a while)")
    data_dict_list = dataset.__getitem__(index=0, start_index=0, end_index=60)
    data_dict_list = to_batch_tensor(data_dict_list)
    logger.info(f"Done preparing data.")
    for i, data_dict in enumerate(data_dict_list):
        # ---- Build input / target dicts ----------------------------------------
        input_dict, target_dict = prepare_inputs_and_targets(data_dict, device)

        # ---- Optionally extend with extrapolated future frames -----------------
        if args.num_extra_frames > 0:
            n = args.num_extra_frames

            # Extend target camera poses (linear-velocity extrapolation)
            input_dict["target_camtoworlds"] = extrapolate_camtoworlds(
                input_dict["target_camtoworlds"], n
            )

            # Extend target intrinsics: repeat the last intrinsics for future frames
            input_dict["target_intrinsics"] = torch.cat(
                [input_dict["target_intrinsics"],
                 input_dict["target_intrinsics"][:, -1:].expand(-1, n, -1, -1, -1)],
                dim=1,
            )

            # Extend target time with constant step extrapolation
            if "target_time" in input_dict:
                dt = (input_dict["target_time"][:, -1:, :]
                      - input_dict["target_time"][:, -2:-1, :])
                extra_t = torch.cat(
                    [input_dict["target_time"][:, -1:, :] + (k + 1) * dt
                     for k in range(n)], dim=1
                )
                input_dict["target_time"] = torch.cat(
                    [input_dict["target_time"], extra_t], dim=1
                )

            # Extend GT images with black frames so the video loop covers all frames
            B, T, V, C, H, W = target_dict["target_image"].shape
            black = torch.zeros(B, n, V, C, H, W,
                                dtype=target_dict["target_image"].dtype,
                                device=target_dict["target_image"].device)
            target_dict["target_image"] = torch.cat(
                [target_dict["target_image"], black], dim=1
            )

            # Extend depth / flow / sky_mask with zeros for extra frames
            for key in ("target_depth", "target_flow", "target_sky_masks"):
                if key in target_dict:
                    t_shape = target_dict[key].shape  # [B, T, V, ...]
                    pad = torch.zeros(
                        t_shape[0], n, *t_shape[2:],
                        dtype=target_dict[key].dtype,
                        device=target_dict[key].device,
                    )
                    target_dict[key] = torch.cat([target_dict[key], pad], dim=1)

            # Extend frame indices
            last_idx = target_dict["target_frame_idx"][:, -1:]  # [B, 1]
            extra_idx = last_idx + torch.arange(
                1, n + 1, device=last_idx.device
            ).unsqueeze(0)
            target_dict["target_frame_idx"] = torch.cat(
                [target_dict["target_frame_idx"], extra_idx], dim=1
            )

        # ---- Compute Gaussian scene representation (shared across renders) -----
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gs_params = model.get_gs_params(input_dict)

        # ---- Save gs_params checkpoint ------------------------------------------
        ckpt_path = os.path.join(checkpoint_dir, f"gs_params_{i}.pth")
        torch.save({k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in gs_params.items()}, ckpt_path)
        logger.info(f"Saved gs_params to {ckpt_path}")

        # ---- Build full-resolution render targets -------------------------------
        scene_id = int(data_dict["scene_id"])
        scene_json = next((ann for ann in dataset.annotations if int(ann["scene_id"]) == scene_id), dataset.annotations[0])
        full_render_dict, camera_list, full_h, full_w = _build_fullres_render_dict(
            input_dict, scene_json, args.num_max_cameras, device
        )
        target_frame_indices = [int(x) for x in target_dict["target_frame_idx"][0].detach().cpu().tolist()]
        gt_fullres = _load_fullres_gt_frames(
            scene_json, args.data_root, target_frame_indices, camera_list, full_h, full_w
        )

        fps = data_dict["fps"] if "fps" in data_dict else 10
        T_tgt = len(target_frame_indices)
        V = len(camera_list)

        for v in range(V):
            cam_render_dict = _clone_render_dict(full_render_dict)
            cam_render_dict["target_camtoworlds"] = full_render_dict["target_camtoworlds"][:, :, v:v+1]
            cam_render_dict["target_intrinsics"] = full_render_dict["target_intrinsics"][:, :, v:v+1]
            if "target_time" in full_render_dict:
                cam_render_dict["target_time"] = full_render_dict["target_time"][:, :, v:v+1]

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                pred_dict = render_chunked(model, gs_params, cam_render_dict, chunk_size=1)

            render_key = pred_dict["render_results"]["rgb_key"]
            pred_images = _denorm_rendered_images(pred_dict["render_results"][render_key][0], device)
            rgb_frames = [_chw_to_uint8(pred_images[t, 0]) for t in range(T_tgt)]
            gt_frames = [gt_fullres[t][v] for t in range(T_tgt)]

            gt_path = os.path.join(video_dir, f"seg{i:02d}_cam{v}_gt_rgb.mp4")
            rgb_path = os.path.join(video_dir, f"seg{i:02d}_cam{v}_rgb.mp4")
            imageio.mimwrite(gt_path, gt_frames, fps=fps, macro_block_size=None)
            imageio.mimwrite(rgb_path, rgb_frames, fps=fps, macro_block_size=None)
            del pred_dict, pred_images, rgb_frames
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if args.render_elevated:
                cam_render_dict_elev = _clone_render_dict(cam_render_dict)
                cam_render_dict_elev["target_camtoworlds"] = elevate_camtoworlds(
                    cam_render_dict["target_camtoworlds"],
                    height=args.elevated_height,
                    tilt_deg=args.elevated_tilt_deg,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred_dict_elev = render_chunked(model, gs_params, cam_render_dict_elev, chunk_size=1)

                elev_key = pred_dict_elev["render_results"]["rgb_key"]
                elev_images = _denorm_rendered_images(pred_dict_elev["render_results"][elev_key][0], device)
                elev_frames = [_chw_to_uint8(elev_images[t, 0]) for t in range(T_tgt)]
                elev_path = os.path.join(video_dir, f"seg{i:02d}_cam{v}_novel_rgb.mp4")
                imageio.mimwrite(elev_path, elev_frames, fps=fps, macro_block_size=None)
                del pred_dict_elev, elev_images, elev_frames, cam_render_dict_elev
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        logger.info(f"Saved gt_rgb / rgb / novel_rgb videos for segment {i} at {full_w}x{full_h}")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
