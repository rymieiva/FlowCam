# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_hydrant.pt" --render_imgs --n_skip 1 --save_imgs
# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\re10k.pt" --render_imgs --n_skip 1 --save_imgs
# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\kitti.pt" --render_imgs --n_skip 1 --save_imgs
# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_10cat.pt" --render_imgs --save_imgs
from run import *

# Evaluation script
import piqa,lpips
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
lpips,psnr,ate=0,0,0

eval_dir = save_dir+"/"+args.name+datetime.datetime.now().strftime("%b%d%Y_")+str(random.randint(0,1e3))
try: os.mkdir(eval_dir)
except: pass
torch.set_grad_enabled(False)

model.n_samples=128

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from data.flowcam_data import FlowCamDataset

val_dataset = FlowCamDataset(
    num_context=2,   # Number of context frames
    n_skip=1,        # Number of frames to skip
    num_trgt=2,      # Number of target frames
    low_res=(128, 128),#(126, 224),  # Resolution for downsampling images
    image_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\Images\0',  # Path to underwater images
    pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',
)


for eval_idx,eval_dataset_idx in enumerate(tqdm(torch.linspace(0,len(val_dataset)-1,min(args.n_eval,len(val_dataset))).int())):
    model_input,ground_truth = val_dataset[eval_dataset_idx]
    # print(f"Model Input Keys: {model_input.keys()}")
    # print(f"Target Poses: {model_input['trgt_c2w']}")
    # print(f"Context Poses: {model_input['ctxt_c2w']}")

    for x in (model_input,ground_truth): 
        for k,v in x.items(): x[k] = v[None].cuda() # collate

    model_out = model.render_full_img(model_input)

    # remove last frame since used as ctxt when n_ctxt=2
    rgb_est,rgb_gt = [rearrange(img[:,:-1].clip(0,1),"b trgt (x y) c -> (b trgt) c x y",x=model_input["trgt_rgb"].size(-2)) 
                                            for img in (model_out["fine_rgb" if "fine_rgb" in model_out else "rgb"],ground_truth["trgt_rgb"])]
    depth_est = rearrange(model_out["depth"][:,:-1],"b trgt (x y) c -> (b trgt) c x y",x=model_input["trgt_rgb"].size(-2))

    psnr += piqa.PSNR()(rgb_est.clip(0,1).contiguous(),rgb_gt.clip(0,1).contiguous())
    lpips += loss_fn_vgg(rgb_est*2-1,rgb_gt*2-1).mean()

    print(args.save_imgs)
    if args.save_imgs:
        fp = os.path.join(eval_dir,f"{eval_idx}_est.png");plt.imsave(fp,make_grid(rgb_est).permute(1,2,0).clip(0,1).cpu().numpy())
        if depth_est.size(1)==3: fp = os.path.join(eval_dir,f"{eval_idx}_depth.png");plt.imsave(fp,make_grid(depth_est).clip(0,1).permute(1,2,0).cpu().numpy())
        fp = os.path.join(eval_dir,f"{eval_idx}_gt.png");plt.imsave(fp,make_grid(rgb_gt).permute(1,2,0).cpu().numpy())
        print(fp)


    if args.save_imgs and args.save_ind: # save individual images separately
        eval_idx_dir = os.path.join(eval_dir,f"dir_{eval_idx}")

        try: os.mkdir(eval_idx_dir)
        except: pass
        ctxt_rgbs = torch.cat((model_input["ctxt_rgb"][:,0],model_input["trgt_rgb"][:,model_input["trgt_rgb"].size(1)//2],model_input["trgt_rgb"][:,-1]))*.5+.5
        fp = os.path.join(eval_idx_dir,f"ctxt0.png");plt.imsave(fp,ctxt_rgbs[0].clip(0,1).permute(1,2,0).cpu().numpy())
        fp = os.path.join(eval_idx_dir,f"ctxt1.png");plt.imsave(fp,ctxt_rgbs[1].clip(0,1).permute(1,2,0).cpu().numpy())
        fp = os.path.join(eval_idx_dir,f"ctxt2.png");plt.imsave(fp,ctxt_rgbs[2].clip(0,1).permute(1,2,0).cpu().numpy())
        for i,(rgb_est,rgb_gt,depth) in enumerate(zip(rgb_est,rgb_gt,depth_est)):
            fp = os.path.join(eval_idx_dir,f"{i}_est.png");plt.imsave(fp,rgb_est.clip(0,1).permute(1,2,0).cpu().numpy())
            print(fp)
            fp = os.path.join(eval_idx_dir,f"{i}_gt.png");plt.imsave(fp,rgb_gt.clip(0,1).permute(1,2,0).cpu().numpy())
            if depth_est.size(1)==3: fp = os.path.join(eval_idx_dir,f"{i}_depth.png");plt.imsave(fp,depth.permute(1,2,0).cpu().clip(1e-4,1-1e-4).numpy())

    # Pose plotting/evaluation
    if "poses" in model_out:
        import scipy.spatial
        from matplotlib.lines import Line2D

        # Extract and align poses
        pose_est, pose_gt = (
            model_out["poses"][0][:, :3, -1].cpu(),
            model_input["trgt_c2w"][0][:, :3, -1].cpu(),
        )

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

        # Remove homogeneous coordinate for visualization
        pose_gt = pose_gt_global[:3].T  # Nx3
        pose_est = pose_est_global[:3].T  # Nx3

        # Align the estimated pose with the ground truth
        pose_gt, pose_est, _ = scipy.spatial.procrustes(pose_gt.numpy(), pose_est.numpy())

        # Calculate ATE (Absolute Trajectory Error)
        ate += ((pose_est - pose_gt) ** 2).mean()

        # Save and plot if required
        if args.save_imgs:
            fig = plt.figure(figsize=(12, 10))  # Larger figure for better visibility
            ax = fig.add_subplot(111, projection="3d")

            # Plot ground truth
            ax.plot(
                *pose_gt.T, label="Ground Truth", color="blue", linewidth=2, marker="o", markersize=5
            )
            # Plot estimated
            ax.plot(
                *pose_est.T, label="Estimated", color="orange", linewidth=2, marker="x", markersize=5
            )

            # Annotate each pose with tensor values (offset text slightly for readability)
            for i in range(pose_gt.shape[0]):
                offset = 0.02  # Offset to separate text from points
                ax.text(
                    pose_gt[i, 0] + offset,
                    pose_gt[i, 1] + offset,
                    pose_gt[i, 2] + offset,
                    f"GT: {pose_gt[i]}",
                    color="blue",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="blue"),
                )
                ax.text(
                    pose_est[i, 0] - offset,
                    pose_est[i, 1] - offset,
                    pose_est[i, 2] - offset,
                    f"EST: {pose_est[i]}",
                    color="orange",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.5, edgecolor="orange"),
                )

            # Add metrics and legend
            ax.set_title(f"Pose Comparison\nATE: {ate:.6f}", fontsize=16)
            ax.legend(
                loc="upper left",
                handles=[
                    Line2D([0], [0], color="blue", lw=2, label="Ground Truth"),
                    Line2D([0], [0], color="orange", lw=2, label="Estimated"),
                ],
            )
            ax.set_xlabel("X", fontsize=12)
            ax.set_ylabel("Y", fontsize=12)
            ax.set_zlabel("Z", fontsize=12)
            ax.view_init(elev=20.0, azim=45)

            # Save full trajectory plot
            fp = os.path.join(eval_dir, f"{eval_idx}_pose_plot.png")
            plt.tight_layout()
            plt.savefig(fp, bbox_inches="tight")
            plt.close()


    print(f"psnr {psnr/(1+eval_idx)}, lpips {lpips/(1+eval_idx)}, ate {(ate/(1+eval_idx))**.5}, eval_idx {eval_idx}", flush=True)
    #print(f"psnr {psnr/(1+eval_idx)}, lpips {lpips/(1+eval_idx)}, eval_idx {eval_idx}", flush=True)

