# note for davis dataloader later: temporally consistent depth estimator: https://github.com/yu-li/TCMonoDepth
# note for cool idea of not even downloading data and just streaming from youtube:https://gist.github.com/Mxhmovd/41e7690114e7ddad8bcd761a76272cc3
import matplotlib.pyplot as plt; 
import cv2
import os
import multiprocessing as mp
import torch.nn.functional as F
import torch
import random
import imageio
import numpy as np
from glob import glob
from collections import defaultdict
from pdb import set_trace as pdb
from itertools import combinations
from random import choice
import matplotlib.pyplot as plt
import imageio.v3 as iio
import yaml
from torchvision import transforms
import datetime
import sys
import re
from glob import glob
import os
import gzip
import json
import numpy as np

from PIL import Image
def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth
def _load_depth(path, scale_adjustment) -> np.ndarray:
    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel

# Geometry functions below used for calculating depth, ignore
def glob_imgs(path):
    imgs = []
    for ext in ["*.png", "*.jpg", "*.JPEG", "*.JPG"]:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def parse_intrinsics(intrinsics):
    fx = intrinsics[..., 0, :1]
    fy = intrinsics[..., 1, 1:2]
    cx = intrinsics[..., 0, 2:3]
    cy = intrinsics[..., 1, 2:3]
    return fx, fy, cx, cy


from einops import rearrange, repeat
ch_sec = lambda x: rearrange(x,"... c x y -> ... (x y) c")
hom = lambda x, i=-1: torch.cat((x, torch.ones_like(x.unbind(i)[0].unsqueeze(i))), i)


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    """

    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    """
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if homogeneous:
        return torch.stack((x_lift, y_lift, z, torch.ones_like(z).to(x.device)), dim=-1)
    else:
        return torch.stack((x_lift, y_lift, z), dim=-1)


def world_from_xy_depth(xy, depth, cam2world, intrinsics):
    batch_size, *_ = cam2world.shape

    x_cam = xy[..., 0]
    y_cam = xy[..., 1]
    z_cam = depth

    pixel_points_cam = lift(
        x_cam, y_cam, z_cam, intrinsics=intrinsics, homogeneous=True
    )
    world_coords = torch.einsum("b...ij,b...kj->b...ki", cam2world, pixel_points_cam)[
        ..., :3
    ]

    return world_coords


def get_ray_directions(xy, cam2world, intrinsics, normalize=True):
    z_cam = torch.ones(xy.shape[:-1]).to(xy.device)
    pixel_points = world_from_xy_depth(
        xy, z_cam, intrinsics=intrinsics, cam2world=cam2world
    )  # (batch, num_samples, 3)

    cam_pos = cam2world[..., :3, 3]
    ray_dirs = pixel_points - cam_pos[..., None, :]  # (batch, num_samples, 3)
    if normalize:
        ray_dirs = F.normalize(ray_dirs, dim=-1)
    return ray_dirs

from PIL import Image
def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth
def _load_depth(path, scale_adjustment) -> np.ndarray:
    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel

# NOTE currently using CO3D V1 because they switch to NDC cameras in 2. TODO is to make conversion code (different intrinsics), verify pointclouds, and switch. 

class FlowCamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_context=3,
        n_skip=1,
        num_trgt=1,
        low_res=(128, 128),
        depth_scale=1,
        val=False,
        image_path=r'C:\Users\rymi\work\FlowCam\underwater\0',
        pose_path=r'C:\Users\rymi\work\FlowCam\snapshoot269\data.sfm',
        use_pose=True
    ):
        """
        Custom initialization for underwater dataset without frame annotations.

        Args:
            num_context (int): Number of context frames.
            n_skip (int): Number of frames to skip between context/target frames.
            num_trgt (int): Number of target frames to predict.
            low_res (tuple): Resolution to downscale images.
            depth_scale (float): Scale for depth values (not used here).
            val (bool): Whether this is a validation set.
            base_path (str): Path to the image directory.
        """
        self.n_trgt = num_trgt
        self.num_skip = n_skip
        self.low_res = low_res
        self.depth_scale = depth_scale
        self.val = val
        self.image_path = image_path
        self.pose_path = pose_path
        self.use_pose = use_pose

        # Get all image paths
        self.image_files = sorted(
            [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        )
        if len(self.image_files) < num_context + num_trgt:
            raise ValueError("Not enough images in the dataset to form context and target frames.")

        # Parse images and poses once during initialization
        self.images_by_timestamp = self.parse_timestamps_and_images(self.image_path, self.pose_path)
        self.poses_by_timestamp = self.parse_poses(self.pose_path)
        #self.landmarks = self.parse_landmarks(self.pose_path)

        ##print(f"Extracted {len(self.landmarks)} landmarks.")
        #print(self.landmarks[5])  # Print the first landmark to check

        # Get all sorted timestamps
        self.sorted_timestamps = sorted(self.images_by_timestamp.keys())

        # Total number of usable sequences (adjust for context and target frames)
        self.total_num_data = len(self.image_files) - (num_context + num_trgt - 1) * n_skip

        print(f"Initialized dataset with {self.total_num_data} sequences from {image_path}.")
    
    def __len__(self):
        return self.total_num_data

    def parse_landmarks(self, pose_path):
        """
        Parse the data.sfm file to extract landmarks and their 2D observations.
        
        Args:
            sfm_path (str): Path to the data.sfm file.

        Returns:
            landmarks (list): List of dictionaries containing 3D points and 2D observations.
        """
        with open(pose_path, 'r') as f:
            sfm_data = json.load(f)
        
        landmarks = []
        
        for landmark in sfm_data["structure"]:
            landmark_id = int(landmark["landmarkId"])
            X = np.array([float(coord) for coord in landmark["X"]])  # 3D landmark coordinates
            
            observations = []
            for obs in landmark["observations"]:
                x = np.array([float(coord) for coord in obs["x"]])  # 2D image coordinates
                observations.append(x)
            
            landmarks.append({
                "id": landmark_id,
                "X": X,
                "observations": observations
            })
        
        return landmarks


    def correct_path(self, parsed_json):
        """
        Correct the paths in the given parsed JSON object by replacing incorrect segments.
        
        Args:
            parsed_json (dict): JSON object parsed from the data.sfm file.
        """
        def to_forward_slashes(path):
            return path.replace("\\", "/").replace("\\/", "/")
        
        incorrect_fragment_fwd = "C:/data/Kattegat Center Videos BlueROV-27-09-2021/20210927121113492-CH0 ROV Camera Feed/snapshoot245"
        correct_fragment_fwd = "C:/Users/rymi/work/FlowCam/kattegat"

        for view in parsed_json["views"]:
            old_path = view["path"]
            old_path_fwd = to_forward_slashes(old_path)

            # b) Replace the fragment:
            new_path_fwd = old_path_fwd.replace(incorrect_fragment_fwd, correct_fragment_fwd)

            # c) (Optional) convert back to backslashes:
            #    "C:\Users\rymi\work\FlowCam\kattegat\Images\0\D2021..."
            #    If you prefer forward slashes, you can skip this step.
            new_path = new_path_fwd.replace("/", "\\")

            view["path"] = new_path
            #print(f"Old path: {old_path}")
            #print(f"New path: {view['path']}")
            #print("-----")

    def parse_poses(self, pose_path):
        """
        Parse poses from the given pose file and apply path correction.

        Args:
            pose_path (str): Path to the data.sfm file.

        Returns:
            dict: Mapping of timestamps to SE(3) matrices (flattened).
        """
        with open(pose_path, 'r') as f:
            parsed_json = json.load(f)

        # Correct paths in the JSON object
        self.correct_path(parsed_json)

        parsed_poses = {}
        pose_dict = {pose['poseId']: pose['pose'] for pose in parsed_json['poses']}
        for i, view in enumerate(parsed_json['views']):
            #print(f"Final corrected path: {view['path']}")
            timestamp = self.extract_timestamp(view['path'])
            pose_id = view['poseId']
            if pose_id in pose_dict:
                rotation_matrix = np.array(pose_dict[pose_id]['transform']['rotation']).reshape(3, 3)
                translation = np.array(pose_dict[pose_id]['transform']['center']).reshape(3)
                se3_matrix = np.eye(4)
                se3_matrix[:3, :3] = rotation_matrix
                se3_matrix[:3, 3] = translation

                # Log SE(3) matrix for debugging
                # print(f"Pose ID: {pose_id}")
                # print(f"SE(3) Matrix from parse_poses:\n{se3_matrix}\n")
                parsed_poses[timestamp] = se3_matrix.flatten().tolist()
     
        return parsed_poses

    def parse_timestamps_and_images(self, image_path, pose_path):
        data_file = pose_path
        with open(data_file, 'r') as f:
            parsed_json = json.load(f)

        parsed_images = {}
        for view in parsed_json['views']:
            timestamp = self.extract_timestamp(view['path'])
            view_path = view['path']
            # take only file name from view_path
            view_file = view_path.split("\\")[-1]
            parsed_images[timestamp] = os.path.join(image_path, view_file)
        return parsed_images

    def extract_timestamp(self, path):
        filename = path.split("\\")[-1]
        timestamp_str = filename.split("D")[1].split(".")[0]  # Extract the timestamp part
        date_part, time_part = timestamp_str.split("T")
        time_part = time_part.replace("-", ":", 2).replace("-", ".")  # Fix separators
        dt = datetime.datetime.strptime(f"{date_part}T{time_part}", "%Y-%m-%dT%H:%M:%S.%f")
        return dt.replace(tzinfo=datetime.timezone.utc)  # Attach UTC timezone    

    # Modify __getitem__ to directly load underwater images
    def __getitem__(self, idx):
        print(f"Number of timestamps: {len(self.sorted_timestamps)}")

        if idx >= len(self.sorted_timestamps) - self.n_trgt:
            idx = random.randint(0, len(self.sorted_timestamps) - self.n_trgt - 1)

        # Select timestamps for context and target frames
        selected_timestamps = self.sorted_timestamps[idx:idx + self.n_trgt + 1]

        # Load images and poses for the selected timestamps
        imgs = []
        c2w_matrices = []
        for timestamp in selected_timestamps:
            img_path = self.images_by_timestamp[timestamp]
            img = plt.imread(img_path)
            imgs.append(torch.from_numpy(np.copy(img)).float())  # Copy to ensure writability

            # Get corresponding pose or default to identity
            if timestamp in self.poses_by_timestamp:
                c2w_matrix = torch.tensor(self.poses_by_timestamp[timestamp]).view(4, 4).float()
            else:
                print(f"Warning: No pose found for timestamp {timestamp}. Defaulting to identity.")
                #c2w_matrix = torch.eye(4)
            c2w_matrices.append(c2w_matrix)

        # Convert list of poses to a tensor
        c2w_matrices = torch.stack(c2w_matrices)

        # Split into target and context matrices
        trgt_c2w = c2w_matrices[1:]
        ctxt_c2w = c2w_matrices[:-1]

        #print(f"Target Pose Matrix (c2w):\n{trgt_c2w}")
        #print(f"Context Pose Matrix (c2w):\n{ctxt_c2w}")

        # Load YAML file
        with open("C:\\Users\\rymi\\work\\FlowCam\\snapshoot269\\AIDataloader.yml", "r") as file:
            config = yaml.safe_load(file)
        
        # Extract camera parameters
        camera_params = config["camera_params"]
        image_size = (1920,1080)

        # print(f"camera_params = {camera_params}")
        # print(f"image_size = {image_size}")

        focal_length_x = camera_params["focal_length_x"]
        focal_length_y = camera_params["focal_length_y"]
        principal_point_x = image_size[0] / 2
        principal_point_y = image_size[1] / 2

        # Construct intrinsic matrix
        K = np.eye(3)
        K[0, 0] = focal_length_x
        K[1, 1] = focal_length_y
        K[0, 2] = principal_point_x
        K[1, 2] = principal_point_y

        # Normalize intrinsics based on image size
        K_normalized = K.copy()
        K_normalized[0, :] /= image_size[0]
        K_normalized[1, :] /= image_size[1]

        # Expand to have a batch dimension and repeat the matrix for each frame in the batch.
        intrinsics_normalized = torch.from_numpy(K_normalized).float().unsqueeze(0).repeat(len(imgs), 1, 1)

        # Increase img resolution for RAFT
        low_res=self.low_res
        h, w = low_res # (320, 320)
        large_scale=2
        imgs_large = F.interpolate(
            torch.stack([x.permute(2, 0, 1) for x in imgs]),
            size=(int(h * large_scale), int(w * large_scale)),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

        # Process images into model inputs
        imgs = F.interpolate(
            torch.stack([x.permute(2, 0, 1) for x in imgs]),
            size=(self.low_res[0], int(self.low_res[0] * w / h)),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )

        imgs = imgs / 255.0 * 2 - 1  # Normalize to [-1, 1]

        uv = np.mgrid[0:low_res[0], 0:low_res[1]].astype(float).transpose(1, 2, 0)
        uv = torch.from_numpy(np.flip(uv, axis=-1).copy()).long()
        uv = uv/ torch.tensor([low_res[1]-1, low_res[0]-1])  # uv in [0,1]
        uv = uv[None].expand(len(imgs),-1,-1,-1).flatten(1,2)

        model_input = {
            "trgt_rgb": imgs[1:],
            "ctxt_rgb": imgs[:-1],
            "trgt_rgb_large": imgs_large[1:],
            "ctxt_rgb_large": imgs_large[:-1],
            "trgt_c2w": trgt_c2w,
            "ctxt_c2w": ctxt_c2w,
            "intrinsics": intrinsics_normalized[1:],
            "x_pix": uv[1:],
        }
        gt = {
            "trgt_rgb": ch_sec(imgs[1:])*.5+.5,
            "ctxt_rgb": ch_sec(imgs[:-1])*.5+.5,
            "intrinsics": intrinsics_normalized[1:],
            "x_pix": uv[1:],
        }

        return model_input, gt
