#python rerun_eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_10cat.pt" --render_imgs --save_imgs
import rerun as rr
from run import *
import piqa, lpips
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from data.flowcam_data import FlowCamDataset
import csv
import os
import numpy as np

def save_metrics_to_csv(eval_idx, psnr_value, lpips_value, psnr_file, lpips_file):
    """
    Save PSNR and LPIPS metrics to separate CSV files.

    Args:
        eval_idx (int): The current evaluation step index.
        psnr_value (torch.Tensor or float): The PSNR value to save.
        lpips_value (torch.Tensor or float): The LPIPS value to save.
        psnr_file (str): Path to the PSNR CSV file.
        lpips_file (str): Path to the LPIPS CSV file.
    """

    # Ensure the PSNR file is initialized with a header
    if not os.path.exists(psnr_file):
        with open(psnr_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "PSNR"])

    # Ensure the LPIPS file is initialized with a header
    if not os.path.exists(lpips_file):
        with open(lpips_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "LPIPS"])

    # Extract numerical values from tensors if needed
    if isinstance(psnr_value, torch.Tensor):
        psnr_value = psnr_value.item()
    if isinstance(lpips_value, torch.Tensor):
        lpips_value = lpips_value.item()

    # Save PSNR value
    with open(psnr_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([eval_idx, psnr_value])

    # Save LPIPS value
    with open(lpips_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([eval_idx, lpips_value])

    print(f"Saved metrics for step {eval_idx}: PSNR={psnr_value:.4f}, LPIPS={lpips_value:.4f}")

def save_poses_tum(file_path, timestamps, poses):
    """
    Save poses in TUM format with error handling.

    Args:
        file_path (str): Path to save the file.
        timestamps (list): List of timestamps.
        poses (list): List of 4x4 transformation matrices.
    """
    try:
        # Check if timestamps and poses have the same length
        if len(timestamps) != len(poses):
            raise ValueError("The length of timestamps and poses must be the same.")

        with open(file_path, 'w') as f:
            for i, (ts, pose) in enumerate(zip(timestamps, poses)):
                try:
                    # Extract translation
                    tx, ty, tz = pose[:3, -1]
                except Exception as e:
                    print(f"Failed to extract translation at index {i}: {e}")

                # Convert rotation matrix to quaternion
                try:
                    qx, qy, qz, qw = rotation_matrix_to_quaternion(pose[:3, :3])
                except Exception as e:
                    print(f"Failed to convert rotation matrix to quaternion at index {i}: {e}")

                # Write to file
                f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

    except Exception as e:
        print(f"Error saving poses to '{file_path}': {e}")

def rotation_matrix_to_quaternion(R):
    """
    Convert a 3x3 rotation matrix to a quaternion (qx, qy, qz, qw) with error handling.

    Args:
        R (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: Quaternion (qx, qy, qz, qw).
    """

    from scipy.spatial.transform import Rotation as Rscipy
    return Rscipy.from_matrix(R).as_quat()  # Returns (qx, qy, qz, qw)

# Initialize Rerun
rr.init("FlowCam Evaluation", spawn=True)

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
lpips, psnr, ate = 0, 0, 0

# Evaluation directory
eval_dir = save_dir + "/" + args.name + datetime.datetime.now().strftime("%b%d%Y_") + str(random.randint(0, 1e3))
os.makedirs(eval_dir, exist_ok=True)

torch.set_grad_enabled(False)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Select experiment
# Dataset
kattegat = 0
okeanos = 0
pipe = 1
# Experiment type
rerun = 0
evo = 1

#psnr_file="1kattegat_r10k_psnr_values.csv"
#lpips_file="1kattegat_r10k_lpips_values.csv"
tum_gt = "1ground_truth_pipe.tum"
tum_est = "estimated_pipe_r10k.tum"

# psnr_file="pipe_r10k_psnr_values.csv"
# lpips_file="pipe_r10k_lpips_values.csv"
# tum_gt = "ground_truth_pipe_r10k.tum"
# tum_est = "estimated_pipe_r10k.tum"

#TODO: evo gt,est for kattegat kitti ntrgt=1. Fix yt vid stats
#TODO: evo gt, est for kattegat r1-k ntrgt=1.  #DONE fix vid stats.
#TODO: evo gt,est for okeanos kitti and r10k ntgrt=1

if kattegat == True:
    # Load validation dataset
    from data.flowcam_data_katt import FlowCamDataset
    val_dataset = FlowCamDataset(
        num_context=1,
        n_skip=1,
        num_trgt=1,
        low_res=(200, 200),
        image_path=r'C:\Users\rymi\work\FlowCam\kattegat\Images\0',
        pose_path=r'C:\Users\rymi\work\FlowCam\kattegat\data.sfm',
    )
elif okeanos == True:
    from data.flowcam_data import FlowCamDataset
    val_dataset = FlowCamDataset(
        num_context=1,
        n_skip=1,
        num_trgt=1,
        low_res=(200, 200),
        image_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\Images\0',
        pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',
    )
elif pipe == True:
    from data.flowcam_data_pipe import FlowCamDataset
    val_dataset = FlowCamDataset(
        num_context=1,
        n_skip=1,
        num_trgt=1,
        low_res=(200, 200),
        image_path=r'C:\Users\rymi\work\FlowCam\deep_ocean_pipe\Images\0',
        pose_path=r'C:\Users\rymi\work\FlowCam\deep_ocean_pipe\data.json',
    )

# Initialize gt trajectory storage
trajectory_gt = []
# List to store accumulated ground truth 4x4 poses
accumulated_gt_poses = []
# Load the transformation matrix from file
if okeanos == True:
    global_transform = np.loadtxt("C:\\Users\\rymi\\work\FlowCam\\snapshoot269\\to_global_transformation.txt")
elif kattegat == True:
    global_transform = np.loadtxt("C:\\Users\\rymi\\work\\FlowCam\\kattegat\\to_global_transformation.txt")
elif pipe == True:
    global_transform = np.loadtxt("C:\\Users\\rymi\\work\\FlowCam\\deep_ocean_pipe\\to_global_transformation.txt")

# Convert to PyTorch tensor for compatibility
global_transform = torch.tensor(global_transform).float().cpu()

# Initialize tensors for storing poses and rendered images
all_poses = torch.tensor([]).cpu()
unique_indices = torch.unique(torch.linspace(0, len(val_dataset) - 1, min(args.n_eval, len(val_dataset))).int())
for eval_idx, eval_dataset_idx in enumerate(unique_indices):
    
    print(f"eval_idx: {eval_idx}, eval_dataset_idx: {eval_dataset_idx}")
    model_input, ground_truth = val_dataset[eval_dataset_idx]

    for x in (model_input, ground_truth): 
        for k, v in x.items(): 
            x[k] = v[None].cuda()

    with torch.no_grad():
        model_out = model.render_full_img(model_input)

    if rerun == True:
        # Compute RGB and Depth
        rgb_est, rgb_gt = [
            rearrange(img[:, :-1].clip(0, 1), "b trgt (x y) c -> (b trgt) c x y", x=model_input["trgt_rgb"].size(-2)) 
            for img in (model_out["fine_rgb" if "fine_rgb" in model_out else "rgb"], ground_truth["trgt_rgb"])
        ]
        depth_est = rearrange(model_out["depth"][:, :-1], "b trgt (x y) c -> (b trgt) c x y", x=model_input["trgt_rgb"].size(-2))
        # Set a time sequence (optional for time-based visualizations)
        rr.set_time_sequence("evaluation_step", eval_idx)
        
        # # Log Images to Rerun
        rr.log("images/estimated", rr.Image(rgb_est[0].permute(1, 2, 0).cpu().numpy()))
        rr.log("images/ground_truth", rr.Image(rgb_gt[0].permute(1, 2, 0).cpu().numpy()))
        rr.log("images/depth/estimated", rr.Image(depth_est[0].permute(1, 2, 0).cpu().numpy()))

        # Compute PSNR and LPIPS for the current step
        psnr += piqa.PSNR()(rgb_est.clip(0, 1).contiguous(), rgb_gt.clip(0, 1).contiguous())
        lpips += loss_fn_vgg(rgb_est * 2 - 1, rgb_gt * 2 - 1).mean()
        
        # Log scalars
        rr.log("metrics/PSNR", rr.Scalar(psnr.item() / (1 + eval_idx)))
        rr.log("metrics/LPIPS", rr.Scalar(lpips.item() / (1 + eval_idx)))

        # Save metrics to CSV for the current step
        psnr_value = piqa.PSNR()(rgb_est.clip(0, 1).contiguous(), rgb_gt.clip(0, 1).contiguous()).item()
        lpips_value = loss_fn_vgg(rgb_est * 2 - 1, rgb_gt * 2 - 1).mean().item()
        save_metrics_to_csv(eval_idx, psnr_value, lpips_value, psnr_file, lpips_file)
    
    # Visualize Poses
    if "poses" in model_out:

        # Get the current pose transformation
        curr_transfs = model_out["poses"][0].cpu()

        if len(all_poses): 
            curr_transfs = all_poses[[-1]] @ curr_transfs # integrate poses
        all_poses = torch.cat((all_poses,curr_transfs)).cpu()
        
        if rerun == True:
            # Get the ground truth and estimated poses
            pose_gt = model_input["trgt_c2w"][0].cpu()
            pose_gt_translation = pose_gt[:, :3, -1].numpy()
            trajectory_gt.extend(pose_gt_translation.tolist())
            pose_est = all_poses[:, :3, -1].numpy()

            # Log the full trajectories
            rr.log(
                "poses/trajectory/ground_truth",
                rr.LineStrips3D(trajectory_gt, colors=[[0, 0, 255]])  # Blue for ground truth
            )
            # Log the estimated trajectory directly (contains cumulative poses)
            rr.log(
                "poses/trajectory/estimated",
                rr.LineStrips3D(pose_est, colors=[[255, 165, 0]])  # Orange for estimated
            )

            # Extract translation and rotation from global_transform
            translation = global_transform[:3, 3].numpy()  # Last column
            rotation = global_transform[:3, :3].numpy()    # Upper-left 3x3
            
            # Log the transformation
            rr.log(
                "poses/transformation/global_transform",
                rr.Transform3D(translation=translation, mat3x3=rotation)
            )
        else:
            timestamps = [i for i in range(len(all_poses))]
            save_poses_tum(tum_est, timestamps, all_poses)

            pose_gt_4x4_batch = model_input["trgt_c2w"][0].cpu().numpy()

            # Append each 4x4 pose in the batch individually
            for pose_gt_4x4 in pose_gt_4x4_batch:
                accumulated_gt_poses.append(pose_gt_4x4)

            # print(f"pose_gt_4x4:\n{pose_gt_4x4}")

            # For pose_gt_global
            timestamps = [i for i in range(len(accumulated_gt_poses))]
            save_poses_tum(tum_gt, timestamps, accumulated_gt_poses)
            print(f"Successfully saved poses to '{tum_gt}' & '{tum_est}'.")

