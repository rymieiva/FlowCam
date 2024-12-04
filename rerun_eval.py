#python rerun_eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_10cat.pt" --render_imgs --save_imgs
import rerun
print(rerun.__version__)

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
from data.flowcam_data import FlowCamDataset

# Load validation dataset
val_dataset = FlowCamDataset(
    num_context=2,
    n_skip=1,
    num_trgt=2,
    low_res=(128, 128),
    image_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\Images\0',
    pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',
)

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

    # Log Images to Rerun
    rr.log_image("images/estimated", rgb_est[0].permute(1, 2, 0).cpu().numpy())
    rr.log_image("images/ground_truth", rgb_gt[0].permute(1, 2, 0).cpu().numpy())
    if depth_est.size(1) == 3:
        rr.log_image("images/depth", depth_est[0].permute(1, 2, 0).cpu().numpy())

    # Log Metrics
    rr.log_scalar("metrics/PSNR", psnr / (1 + eval_idx))
    rr.log_scalar("metrics/LPIPS", lpips / (1 + eval_idx))

    # Visualize Poses
    if "poses" in model_out:
        pose_est, pose_gt = (
            model_out["poses"][0][:, :3, -1].cpu(),
            model_input["trgt_c2w"][0][:, :3, -1].cpu(),
        )

        # Apply global transformation
        global_transform = np.loadtxt("C:\\Users\\rymi\\work\\FlowCam\\snapshoot269\\to_global_transformation.txt")
        global_transform = torch.tensor(global_transform).float()

        pose_gt_h = torch.cat((pose_gt, torch.ones(pose_gt.size(0), 1)), dim=1).T
        pose_est_h = torch.cat((pose_est, torch.ones(pose_est.size(0), 1)), dim=1).T

       
