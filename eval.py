# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_hydrant.pt" --render_imgs --low_res 144 128 --n_skip 1 --save_imgs
# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\re10k.pt" --render_imgs --low_res 144 128 --n_skip 1 --save_imgs
# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\kitti.pt" --render_imgs --low_res 144 128 --n_skip 1 --save_imgs
# python eval.py -c "C:\Users\rymi\OneDrive - EIVA\Desktop\flowcam\co3d_10cat.pt" --render_imgs --low_res 144 128 --n_skip 1 --save_imgs
from run import *

# Evaluation script
import piqa,lpips
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
lpips,psnr,ate=0,0,0

eval_dir = save_dir+"/"+args.name+datetime.datetime.now().strftime("%b%d%Y_")+str(random.randint(0,1e3))
try: os.mkdir(eval_dir)
except: pass
torch.set_grad_enabled(False)

model.n_samples=128

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from data.flowcam_data import FlowCamDataset
# Instantiation of FlowCamDataset in demo.py
# val_dataset = FlowCamDataset(
#     num_context=6,
#     n_skip=int(args.n_skip),           # Convert n_skip to integer explicitly
#     num_trgt=6,
#     low_res=args.low_res,
#     depth_scale=1,
#     val=True,
#     num_cat=1000,
#     overfit=False,
#     category="hydrant",                # Currently, only hydrant dataset available, set it to be the category
#     use_mask=False,
#     use_v1=True
# )

val_dataset = FlowCamDataset(
    num_context=2,   # Number of context frames
    n_skip=1,        # Number of frames to skip
    num_trgt=2,      # Number of target frames
    low_res=(320, 320),#(126, 224),  # Resolution for downsampling images
    base_path=r'C:\Users\rymi\work\FlowCam\underwater\0',  # Path to underwater images
)


for eval_idx,eval_dataset_idx in enumerate(tqdm(torch.linspace(0,len(val_dataset)-1,min(args.n_eval,len(val_dataset))).int())):
    model_input,ground_truth = val_dataset[eval_dataset_idx]

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
    # if "poses" in model_out:
        # import scipy.spatial
        # pose_est,pose_gt = model_out["poses"][0][:,:3,-1].cpu(),model_input["trgt_c2w"][0][:,:3,-1].cpu()

        # print(f"pose_gt shape: {pose_gt.shape}, unique points: {torch.unique(pose_gt, dim=0).size(0)}")
        # print(f"pose_est shape: {pose_est.shape}, unique points: {torch.unique(pose_est, dim=0).size(0)}")
        # pose_gt,pose_est,_ = scipy.spatial.procrustes(pose_gt.numpy(),pose_est.numpy())

        # pose_est, pose_gt = model_out["poses"][0][:, :3, -1].cpu(), model_input["trgt_c2w"][0][:, :3, -1].cpu()

            # Debug shapes and unique points
        # print(f"pose_gt shape: {pose_gt.shape}, unique points: {torch.unique(pose_gt, dim=0).size(0)}")
        # print(f"pose_est shape: {pose_est.shape}, unique points: {torch.unique(pose_est, dim=0).size(0)}")
            # if torch.unique(pose_gt, dim=0).size(0) <= 1:
            #     pose_gt += torch.randn_like(pose_gt) * 1e-6  # Add small noise to make points unique
            # if torch.unique(pose_est, dim=0).size(0) <= 1:
            #     pose_est += torch.randn_like(pose_est) * 1e-6
            # pose_gt, pose_est, _ = scipy.spatial.procrustes(pose_gt.numpy(), pose_est.numpy())

            # Handle insufficient unique points
        # if torch.unique(pose_gt, dim=0).size(0) > 1 and torch.unique(pose_est, dim=0).size(0) > 1:
        #     pose_gt, pose_est, _ = scipy.spatial.procrustes(pose_gt.numpy(), pose_est.numpy())
        # else:
        #     print(f"Skipping Procrustes analysis due to insufficient unique points. "
        #     f"pose_gt unique points: {torch.unique(pose_gt, dim=0).size(0)}, "
        #     f"pose_est unique points: {torch.unique(pose_est, dim=0).size(0)}")

        # ate += ((pose_est-pose_gt)**2).mean()
        # if args.save_imgs:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     ax.plot(*pose_gt.T)
        #     ax.plot(*pose_est.T)
        #     ax.xaxis.set_tick_params(labelbottom=False)
        #     ax.yaxis.set_tick_params(labelleft=False)
        #     ax.zaxis.set_tick_params(labelleft=False)
        #     ax.view_init(elev=10., azim=45)
        #     plt.tight_layout()
        #     fp = os.path.join(eval_dir,f"{eval_idx}_pose_plot.png");plt.savefig(fp,bbox_inches='tight');plt.close()
        #     if args.save_ind:
        #         for i in range(len(pose_est)):
        #             fig = plt.figure()
        #             ax = fig.add_subplot(111, projection='3d')
        #             ax.plot(*pose_gt.T,color="black")
        #             ax.plot(*pose_est.T,alpha=0)
        #             ax.plot(*pose_est[:i].T,alpha=1,color="red")
        #             ax.xaxis.set_tick_params(labelbottom=False)
        #             ax.yaxis.set_tick_params(labelleft=False)
        #             ax.zaxis.set_tick_params(labelleft=False)
        #             ax.view_init(elev=10., azim=45)
        #             plt.tight_layout()
        #             fp = os.path.join(eval_idx_dir,f"pose_{i}.png"); plt.savefig(fp,bbox_inches='tight');plt.close()

    # print(f"psnr {psnr/(1+eval_idx)}, lpips {lpips/(1+eval_idx)}, ate {(ate/(1+eval_idx))**.5}, eval_idx {eval_idx}", flush=True)
    print(f"psnr {psnr/(1+eval_idx)}, lpips {lpips/(1+eval_idx)}, eval_idx {eval_idx}", flush=True)

