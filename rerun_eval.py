#python rerun_eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_10cat.pt" --render_imgs --save_imgs
import rerun as rr
from run import *
import piqa, lpips
from torchvision.utils import make_grid
import numpy as np
from tqdm import tqdm
from data.flowcam_data import FlowCamDataset

# Initialize Rerun
rr.init("FlowCam Evaluation", spawn=True)

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
lpips, psnr, ate = 0, 0, 0

# Evaluation directory
eval_dir = save_dir + "/" + args.name + datetime.datetime.now().strftime("%b%d%Y_") + str(random.randint(0, 1e3))
os.makedirs(eval_dir, exist_ok=True)

torch.set_grad_enabled(False)
model.n_samples = 128

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# FlowCam author: This code uses an "OpenCV" style camera coordinate system, where the Y-axis points downwards (the up-vector points in the negative Y-direction), the X-axis points right, and the Z-axis points into the image plane.
# Does this convention match with the dataset coordinate system?
def convert_to_opencv_convention(pose):
    # Assumes input pose uses a different convention (e.g., Y-up, Z-forward)
    transform_matrix = torch.tensor([
        [1,  0,  0,  0],  # X remains the same
        [0, -1,  0,  0],  # Flip Y
        [0,  0, -1,  0],  # Flip Z
        [0,  0,  0,  1]   # Homogeneous coordinate
    ], device=pose.device, dtype=pose.dtype)
    return transform_matrix @ pose

# Load validation dataset
# from data.flowcam_data import FlowCamDataset
# val_dataset = FlowCamDataset(
#     num_context=2,
#     n_skip=1,
#     num_trgt=2,
#     low_res=(128, 128),
#     image_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\Images\0',
#     pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',
# )
from data.co3d import Co3DNoCams
val_dataset= Co3DNoCams(
    n_skip=1,
    num_trgt=3,
    num_context=3,
    low_res=(128,128),
    #val=True,
    num_cat=1,
    category="hydrant"
)

# Initialize global trajectory storage
trajectory_gt = []  # Ground truth trajectory
trajectory_est = []  # Estimated trajectory

for eval_idx, eval_dataset_idx in enumerate(tqdm(torch.linspace(0, len(val_dataset)-1, min(args.n_eval, len(val_dataset))).int())):
    model_input, ground_truth = val_dataset[eval_dataset_idx]

    for x in (model_input, ground_truth): 
        for k, v in x.items(): 
            x[k] = v[None].cuda()

    model_out = model.render_full_img(model_input)

    # Compute RGB and Depth
    rgb_est, rgb_gt = [
        rearrange(img[:, :-1].clip(0, 1), "b trgt (x y) c -> (b trgt) c x y", x=model_input["trgt_rgb"].size(-2)) 
        for img in (model_out["fine_rgb" if "fine_rgb" in model_out else "rgb"], ground_truth["trgt_rgb"])
    ]
    depth_est = rearrange(model_out["depth"][:, :-1], "b trgt (x y) c -> (b trgt) c x y", x=model_input["trgt_rgb"].size(-2))

    # Compute Metrics
    psnr += piqa.PSNR()(rgb_est.clip(0, 1).contiguous(), rgb_gt.clip(0, 1).contiguous())
    lpips += loss_fn_vgg(rgb_est * 2 - 1, rgb_gt * 2 - 1).mean()

    # Set a time sequence (optional for time-based visualizations)
    rr.set_time_sequence("evaluation_step", eval_idx)
    
    # Log Images to Rerun
    rr.log("images/estimated", rr.Image(rgb_est[0].permute(1, 2, 0).cpu().numpy()))
    rr.log("images/ground_truth", rr.Image(rgb_gt[0].permute(1, 2, 0).cpu().numpy()))

    if depth_est.size(1) == 3:
        rr.log("images/depth", rr.Image(depth_est[0].permute(1, 2, 0).cpu().numpy()))

    # Log scalars
    rr.log("metrics/PSNR", rr.Scalar(psnr.item() / (1 + eval_idx)))
    rr.log("metrics/LPIPS", rr.Scalar(lpips.item() / (1 + eval_idx)))

    # Visualize Poses
    if "poses" in model_out:
        import scipy.spatial
        pose_est, pose_gt = (
            model_out["poses"][0][:, :3, -1].cpu(),
            model_input["trgt_c2w"][0][:, :3, -1].cpu(),
        )
        # print("\nEstimated Poses:", pose_est)
        # print("GT Poses:\n", pose_gt)
        # pose_gt_np = pose_gt.cpu().numpy()
        # pose_est_np = pose_est.cpu().numpy()

        # rr.log("raw/trajectory/ground_truth", rr.LineStrips3D(pose_gt_np, colors=[[0, 0, 255]]))
        # rr.log("raw/trajectory/estimated", rr.LineStrips3D(pose_est_np, colors=[[255, 165, 0]]))

        # Load the transformation matrix from file
        global_transform = np.loadtxt("C:\\Users\\rymi\\work\FlowCam\\snapshoot269\\to_global_transformation.txt")

        # Convert to PyTorch tensor for compatibility
        global_transform = torch.tensor(global_transform).float()

        # Convert pose_gt and pose_est to homogeneous coordinates if they aren't already
        pose_gt_homogeneous = torch.cat((pose_gt, torch.ones(pose_gt.size(0), 1)), dim=1).T  # 4xN
        pose_est_homogeneous = torch.cat((pose_est, torch.ones(pose_est.size(0), 1)), dim=1).T  # 4xN
        
        # Transform to global frame
        pose_gt_global = global_transform @ pose_gt_homogeneous  # 4xN
        pose_est_global = global_transform @ pose_est_homogeneous  # 4xN

        #pose_gt_opencv = convert_to_opencv_convention(pose_gt_global)
        #pose_est_opencv = convert_to_opencv_convention(pose_est_global)
        #pose_gt = pose_est_opencv[:3].T  # Nx3
        #pose_est = pose_est_opencv[:3].T  # Nx3
        
        pose_gt = pose_gt_global[:3].T  # Nx3
        pose_est = pose_est_global[:3].T  # Nx3
        
        # print("Global Transform:\n", global_transform)
        # print("Pose GT (Transformed):\n", pose_gt_global[:5])  # First 5 ground truth poses
        # print("Pose EST (Transformed):\n", pose_est_global[:5])  # First 5 estimated poses
        # print("Pose GT (No Homogeneuos coordinate):\n", pose_gt[:5])
        # print("Pose GT (No Homogeneuos coordinate):\n", pose_est[:5])
        
        # # Align the estimated pose with the ground truth
        #pose_gt, pose_est, _ = scipy.spatial.procrustes(pose_gt.numpy(), pose_est.numpy())
        
        # Calculate ATE (Absolute Trajectory Error)
        ate += ((pose_est - pose_gt) ** 2).mean()
        
        # Log ATE as a scalar
        rr.log("metrics/ATE", rr.Scalar(ate.item() ** 0.5))

        # Extract translation and rotation from global_transform
        translation = global_transform[:3, 3].numpy()  # Last column
        rotation = global_transform[:3, :3].numpy()    # Upper-left 3x3
        
        # Log the transformation
        rr.log(
            "poses/transformation/global_transform",
            rr.Transform3D(translation=translation, mat3x3=rotation)
        )
        
        # Append new positions to the full trajectory
        trajectory_gt.extend(pose_gt.tolist())
        trajectory_est.extend(pose_est.tolist())

        # # Log full trajectories to Rerun
        rr.log(
            "poses/trajectory/ground_truth",
            rr.LineStrips3D(np.array(trajectory_gt), colors=[[0, 0, 255]])  # Blue for GT
        )
        rr.log(
            "poses/trajectory/estimated",
            rr.LineStrips3D(np.array(trajectory_est), colors=[[255, 165, 0]])  # Orange for EST
        )

        # Log the individual current step's pose
        rr.log(
            f"poses/ground_truth_step/{eval_idx}",
            rr.Points3D(pose_gt, colors=[[0, 0, 255]], radii=[0.05])
        )
        rr.log(
            f"poses/estimated_step/{eval_idx}",
            rr.Points3D(pose_est, colors=[[255, 165, 0]], radii=[0.05])
        )
        # Add axis to poses
        # for i, (gt_pose, est_pose) in enumerate(zip(pose_gt, pose_est)):
        #     # Ground Truth Axes
        #     rr.log(
        #         f"poses/ground_truth/{i}/axes",
        #         rr.Arrows3D(
        #             vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # X, Y, Z axes
        #             origins=[gt_pose] * 3,  # Start each arrow at the pose
        #             colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Red, Green, Blue
        #         )
        #     )

        #     # Estimated Axes
        #     rr.log(
        #         f"poses/estimated/{i}/axes",
        #         rr.Arrows3D(
        #             vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # X, Y, Z axes
        #             origins=[est_pose] * 3,  # Start each arrow at the pose
        #             colors=[[255, 165, 0], [165, 255, 0], [0, 165, 255]],  # Orange, Lime, Cyan
        #         )
        #     )       
