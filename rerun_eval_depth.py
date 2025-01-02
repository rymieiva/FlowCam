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

        print(f"Successfully saved poses to '{file_path}'.")

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


import csv
import os

def save_metrics_to_csv(eval_idx, psnr_value, lpips_value, psnr_file="psnr_values.csv", lpips_file="lpips_values.csv"):
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

def project_landmark(X, pose, K):
    """
    Project a 3D landmark into image space using the camera pose and intrinsics.

    Args:
        X (np.array): 3D landmark coordinates (shape: (3,))
        pose (torch.Tensor): 4x4 camera pose (ground truth).
        K (np.array): 3x3 camera intrinsic matrix.

    Returns:
        projected_point (np.array): 2D image coordinates (shape: (2,))
    """
    # Convert X to homogeneous coordinates (shape: (4,))
    X_h = np.append(X, 1)  # [x, y, z, 1]
    #print(f"X_h shape: {X_h.shape}, X_h: {X_h}")

    # Verify pose shape
    #print(f"Pose shape: {pose.shape}")

    # Apply the camera pose to transform the point to the camera frame
    X_cam = pose @ torch.tensor(X_h).float()
    #print(f"X_cam shape: {X_cam.shape}, X_cam: {X_cam}")

    # Extract the first three elements (shape: (3,))
    X_cam_np = X_cam[:3].numpy()
    #print(f"X_cam_np shape: {X_cam_np.shape}, X_cam_np: {X_cam_np}")

    # Ensure X_cam_np is a 3-element vector
    if X_cam_np.shape != (3,):
        raise ValueError(f"X_cam_np must have shape (3,), but got {X_cam_np.shape}")

    # Apply the intrinsic matrix to project onto the image plane
    x_proj = K @ X_cam_np
    #print(f"x_proj before normalization: {x_proj}")

    # Normalize by the third coordinate to get pixel coordinates
    x_proj /= x_proj[2]
    #print(f"x_proj after normalization: {x_proj}")

    return x_proj[:2]

def render_ground_truth_depth(landmark, pose, K, image_size):
    """
    Render ground truth depth image from a single 3D landmark and ground truth pose.
    
    Parameters:
    - landmark: Single 3D landmark (1 x 3).
    - pose: Ground truth camera pose (4 x 4).
    - K: Intrinsic matrix (3 x 3).
    - image_size: Tuple of (height, width).
    
    Returns:
    - depth_image: Depth image (height x width).
    """
    height, width = image_size
    depth_image = np.full((height, width), np.inf)

    # Convert landmark to homogeneous coordinates
    landmark_h = np.append(landmark, 1)  # Shape: (4,)

    # Transform landmark to the camera frame using ground truth pose
    landmark_cam = pose @ landmark_h  # Shape: (4,)

    # Extract the 3D point (X, Y, Z) in the camera frame
    point_3d = landmark_cam[:3]

    # Check if the point is in front of the camera
    if point_3d[2] > 0:
        # Project the 3D point onto the 2D image plane
        point_2d_h = K @ point_3d  # Shape: (3,)
        point_2d = point_2d_h[:2] / point_2d_h[2]  # Normalize by depth

        # Round to nearest integer to get pixel coordinates
        pixel = np.round(point_2d).astype(int)

        # Insert depth value into the depth image if within image bounds
        x, y = pixel
        if 0 <= x < width and 0 <= y < height:
            depth_image[y, x] = point_3d[2]

    # Replace inf values with 0 for visualization
    depth_image[depth_image == np.inf] = 0

    #print(f"3D Landmark: {landmark}")
    #print(f"Transformed 3D Point (Camera Frame): {point_3d}")

    return depth_image

# Load validation dataset
from data.flowcam_data import FlowCamDataset
# val_dataset = FlowCamDataset(
#     num_context=2,
#     n_skip=1,
#     num_trgt=2,
#     low_res=(200, 200),
#     image_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\Images\0',
#     pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',
# )

val_dataset = FlowCamDataset(
    num_context=2,
    n_skip=1,
    num_trgt=2,
    low_res=(200, 200),
    #image_path=r'C:\Users\rymi\work\FlowCam\deep_ocean_pipe\Images\0',
    #pose_path=r'C:\Users\rymi\work\FlowCam\deep_ocean_pipe\data.json',
    image_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\Images\0',
    pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',

)

# from data.co3d import Co3DNoCams
# val_dataset= Co3DNoCams(
#     n_skip=1,
#     num_trgt=2,
#     num_context=2,
#     low_res=(128,128),
#     #val=True,
#     num_cat=1,
#     category="hydrant"
# )

# Initialize global trajectory storage
trajectory_gt = []
trajectory_est = []
# List to store accumulated ground truth 4x4 poses
accumulated_gt_poses = []
# Load the transformation matrix from file
global_transform = np.loadtxt("C:\\Users\\rymi\\work\FlowCam\\snapshoot269\\to_global_transformation.txt")

# Convert to PyTorch tensor for compatibility
global_transform = torch.tensor(global_transform).float().cpu()

# Initialize the starting ground truth pose as the identity matrix
start_gt_pose = None
# List to store relative GT displacements
relative_gt_poses = []

# Initialize tensors for storing poses and rendered images
all_poses = torch.tensor([]).cpu()
unique_indices = torch.unique(torch.linspace(0, len(val_dataset) - 1, min(args.n_eval, len(val_dataset))).int())
for eval_idx, eval_dataset_idx in enumerate(unique_indices):
    model_input, ground_truth = val_dataset[eval_dataset_idx]

    for x in (model_input, ground_truth): 
        for k, v in x.items(): 
            x[k] = v[None].cuda()

    with torch.no_grad():
        model_out = model.render_full_img(model_input)

    # Compute RGB and Depth
    rgb_est, rgb_gt = [
        rearrange(img[:, :-1].clip(0, 1), "b trgt (x y) c -> (b trgt) c x y", x=model_input["trgt_rgb"].size(-2)) 
        for img in (model_out["fine_rgb" if "fine_rgb" in model_out else "rgb"], ground_truth["trgt_rgb"])
    ]
    depth_est = rearrange(model_out["depth"][:, :-1], "b trgt (x y) c -> (b trgt) c x y", x=model_input["trgt_rgb"].size(-2))

    K = model_input["intrinsics"][0][0].cpu().numpy()  # Shape: (3, 3)

    print(f"Intrinsic Matrix K:\n{K}")
    pose_gt = model_input["trgt_c2w"][0][0].cpu().numpy()  # Select the first pose, shape: (4, 4)
    print(f"Ground Truth Pose:\n{pose_gt}")
    # Create an empty ground truth depth map
    #depth_gt = np.zeros(depth_est.shape[2:], dtype=np.float32)
    image_size = depth_est.shape[2:]  # Use the size from depth_est for consistency
    depth_gt = np.full(image_size, np.inf, dtype=np.float32)
    
    for landmark in val_dataset.landmarks:
        X = landmark["X"]  # 3D point
        single_depth = render_ground_truth_depth(X, pose_gt, K, image_size)
        depth_gt = np.minimum(depth_gt, single_depth)

    # Replace inf values with 0 for visualization
    depth_gt[depth_gt == np.inf] = 0

    if np.any(depth_gt < np.inf):
        valid_depths = depth_gt[depth_gt < np.inf]
        depth_gt_normalized = (depth_gt - np.min(valid_depths)) / (np.max(valid_depths) - np.min(valid_depths))
    else:
        depth_gt_normalized = np.zeros_like(depth_gt)  # No valid depth values

    # Log the normalized ground truth depth
    rr.log("images/depth/ground_truth_normalized", rr.Image(depth_gt_normalized))
    
    rr.log("images/depth/ground_truth", rr.Image(depth_gt))

    # Ensure depth_gt has the correct shape (H, W)
    if len(depth_gt.shape) == 2:
        depth_gt = depth_gt[:, :, np.newaxis]  # Expand to (H, W, 1)

    rr.log("images/depth/ground_truth_expanded", rr.Image(depth_gt))
    # Log the depth maps
    rr.log("images/depth/estimated", rr.Image(depth_est[0].permute(1, 2, 0).cpu().numpy()))

    # Compute PSNR and LPIPS for the current step
    psnr_value = piqa.PSNR()(rgb_est.clip(0, 1).contiguous(), rgb_gt.clip(0, 1).contiguous()).item()
    lpips_value = loss_fn_vgg(rgb_est * 2 - 1, rgb_gt * 2 - 1).mean().item()

    # Save metrics to CSV for the current step
    save_metrics_to_csv(eval_idx, psnr_value, lpips_value)

    # # Set a time sequence (optional for time-based visualizations)
    # rr.set_time_sequence("evaluation_step", eval_idx)
    
    # # Log Images to Rerun
    rr.log("images/estimated", rr.Image(rgb_est[0].permute(1, 2, 0).cpu().numpy()))
    rr.log("images/ground_truth", rr.Image(rgb_gt[0].permute(1, 2, 0).cpu().numpy()))


    
    #if depth_est.size(1) == 3:
        #rr.log("images/depth/estimated", rr.Image(depth_est[0].permute(1, 2, 0).cpu().numpy()))
    #rr.log("images/depth/ground_truth", rr.Image(depth_gt[0].permute(1, 2, 0).cpu().numpy()))
    # # Log scalars
    # rr.log("metrics/PSNR", rr.Scalar(psnr.item() / (1 + eval_idx)))
    # rr.log("metrics/LPIPS", rr.Scalar(lpips.item() / (1 + eval_idx)))
    # Visualize Poses
    if "poses" in model_out:
        #print(f"eval_idx: {eval_idx}, eval_dataset_idx: {eval_dataset_idx}")

        # Get the current pose transformation
        curr_transfs = model_out["poses"][0].cpu()
        #print(f"curr_transfs shape: {curr_transfs.shape}")

        if len(all_poses): 
            curr_transfs = all_poses[[-1]] @ curr_transfs # integrate poses
        all_poses = torch.cat((all_poses,curr_transfs)).cpu()
        all_poses = global_transform @ all_poses
        pose_est = all_poses[:, :3, -1].numpy()
        #print(f"all_poses shape: {all_poses.shape}")
        #all_poses = global_transform @ all_poses
        # For pose_est_global_acc
        timestamps = [i for i in range(len(all_poses))]
        save_poses_tum("estimated_pipe_10cat.tum", timestamps, all_poses)

        # Get the ground truth and estimated poses
        # pose_gt = model_input["trgt_c2w"][0].cpu()
        # pose_gt = global_transform @ pose_gt
        # pose_gt_translation = pose_gt[:, :3, -1].numpy()
        # trajectory_gt.extend(pose_gt_translation.tolist())
    
        pose_gt_4x4_batch = model_input["trgt_c2w"][0].cpu().numpy()
        pose_gt_4x4_batch = global_transform @ pose_gt_4x4_batch

        # Append each 4x4 pose in the batch individually
        for pose_gt_4x4 in pose_gt_4x4_batch:
           accumulated_gt_poses.append(pose_gt_4x4)

        #print(f"pose_gt_4x4:\n{pose_gt_4x4}")

        # For pose_gt_global
        timestamps = [i for i in range(len(accumulated_gt_poses))]
        save_poses_tum("ground_truth_pipe_10cat.tum", timestamps, accumulated_gt_poses)


    #     # pose_est_translated_acc = all_poses[:, :3, -1].numpy()
        
    #     # #pose_est_translated_acc = all_poses[:, :3, -1].numpy()

    #     # # Log the individual current step's pose
    #     # rr.log(
    #     #     f"poses/ground_truth_step/{eval_idx}",
    #     #     rr.Points3D(pose_gt, colors=[[0, 0, 255]], radii=[0.05])
    #     # )
    #     # # rr.log(
    #     # #     f"poses/estimated_step/{eval_idx}",
    #     # #     rr.Points3D(pose_est_translated_acc, colors=[[255, 165, 0]], radii=[0.05])
    #     # # )

    #     # Log the full trajectories
    #     rr.log(
    #         "poses/trajectory/ground_truth",
    #         rr.LineStrips3D(trajectory_gt, colors=[[0, 0, 255]])  # Blue for ground truth
    #     )
    #     # Log the estimated trajectory directly (contains cumulative poses)
    #     rr.log(
    #         "poses/trajectory/estimated",
    #         rr.LineStrips3D(pose_est, colors=[[255, 165, 0]])  # Orange for estimated
    #     )


    #     # pose_est_translated_acc = torch.tensor(pose_est_translated_acc)  # Convert to PyTorch tensor

    #     # #print(f"pose_est_translated_acc = {pose_est_translated_acc}\n")
    #     # # Slice the latest rows to match the shape of the current ground truth pose
    #     # pose_est_translated_acc_latest = pose_est_translated_acc[-pose_gt.size(0):]

    #     # # print(f"pose_gt = {pose_gt}\n")
    #     # # print(f"pose_est_translated_acc_latest = {pose_est_translated_acc_latest}\n")
    #     # # Calculate ATE (Absolute Trajectory Error)
    #     # ate += ((pose_est_translated_acc_latest - pose_gt) ** 2).mean()
        
    #     # # Log ATE as a scalar
    #     # rr.log("metrics/ATE", rr.Scalar(ate.item() ** 0.5))
    #     # rr.log("ATE", rr.Scalar(ate.item() ** 0.5))
    #     # # Extract translation and rotation from global_transform
    #     # translation = global_transform[:3, 3].numpy()  # Last column
    #     # rotation = global_transform[:3, :3].numpy()    # Upper-left 3x3
        
    #     # # Log the transformation
    #     # rr.log(
    #     #     "poses/transformation/global_transform",
    #     #     rr.Transform3D(translation=translation, mat3x3=rotation)
    #     # )

    #     # curr_gt_pose = model_input["trgt_c2w"][0].cpu()  # Shape: (N, 4, 4)
        
    #     # # Initialize the starting ground truth pose on the first iteration
    #     # if start_gt_pose is None:
    #     #     start_gt_pose = curr_gt_pose.clone()

    #     # # Compute the relative ground truth pose with respect to the starting pose
    #     # displacement = curr_gt_pose - start_gt_pose

    #     # # Store the relative displacement
    #     # relative_gt_poses.append(displacement)

    #     # # Update the cumulative previous GT pose
    #     # rel_gt_pose = displacement  # Equivalent to setting cumulative_prev_gt_pose = curr_gt_pose
    
    #     # # Convert to homogeneous coordinates
    #     # rel_gt_pose_homogeneous = torch.cat((rel_gt_pose[:, :3, -1], torch.ones(rel_gt_pose.size(0), 1)), dim=1).T  # Shape: (4, N)
    #     # # Transform to global frame
    #     # rel_gt_pose_global = global_transform @ rel_gt_pose_homogeneous
    #     # rel_gt_pose = rel_gt_pose_global[:3].T