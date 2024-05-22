import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from utils.metrics_utils import calculate_au_pro
from utils.pointnet2_utils import interpolating_points
from models.full_models import FeatureExtractors


dino_backbone_name = 'vit_base_patch8_224.dino' # 224/8 -> 28 patches.
group_size = 128
num_group = 1024

class MultimodalFeatures(torch.nn.Module):
    def __init__(self, image_size = 224):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.deep_feature_extractor = FeatureExtractors(device = self.device, 
                                                 rgb_backbone_name = dino_backbone_name, 
                                                 group_size = group_size, num_group = num_group)

        self.deep_feature_extractor.to(self.device)

        self.image_size = image_size

        # * Applies a 2D adaptive average pooling over an input signal composed of several input planes. 
        # * The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
        self.resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        
        self.average = torch.nn.AvgPool2d(kernel_size = 3, stride = 1) 

    def __call__(self, rgb, xyz):
        rgb = rgb.to(self.device)
        xyz = xyz.to(self.device)

        with torch.no_grad():
            rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx = self.deep_feature_extractor(rgb, xyz)


        interpolated_feature_maps = interpolating_points(xyz, center.permute(0,2,1), xyz_feature_maps)

        xyz_feature_maps = [fmap for fmap in [xyz_feature_maps]]
        rgb_feature_maps = [fmap for fmap in [rgb_feature_maps]]

        return rgb_feature_maps, xyz_feature_maps, center, ori_idx, center_idx, interpolated_feature_maps

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def get_features_maps(self, rgb, pc):

        unorganized_pc = pc.squeeze().permute(1, 2, 0).reshape(-1, pc.shape[1])

        # Find nonzero indices.
        nonzero_indices = torch.nonzero(torch.all(unorganized_pc != 0, dim=1)).squeeze(dim=1)

        # Select nonzero indices and discard the others.
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :].unsqueeze(dim=0).permute(0, 2, 1)

        rgb_feature_maps, xyz_feature_maps, center, neighbor_idx, center_idx, interpolated_pc = self(rgb, unorganized_pc_no_zeros.contiguous())
                
        # Interpolation to obtain a "full image" with point cloud features.
        xyz_patch = torch.cat(xyz_feature_maps, 1)

        xyz_patch_full = torch.zeros((1, interpolated_pc.shape[1], self.image_size * self.image_size), dtype = xyz_patch.dtype, device = self.device)
        xyz_patch_full[..., nonzero_indices] = interpolated_pc
        
        xyz_patch_full_2d = xyz_patch_full.view(1, interpolated_pc.shape[1], self.image_size, self.image_size)
        xyz_patch_full_resized = self.resize(self.average(xyz_patch_full_2d))
        xyz_patch = xyz_patch_full_resized.reshape(xyz_patch_full_resized.shape[1], -1).T 

        rgb_patch = torch.cat(rgb_feature_maps, 1)

        upsample_shape = xyz_patch_full_resized.shape[-2:]
        rgb_patch_upsample = torch.nn.functional.interpolate(rgb_patch, size = upsample_shape, mode = 'bilinear', align_corners = False)
        rgb_patch_upsample = rgb_patch_upsample.reshape(rgb_patch.shape[1], -1).T

        return rgb_patch_upsample, xyz_patch