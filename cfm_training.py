import argparse

import os
import torch
import wandb

import numpy as np
from itertools import chain

from tqdm import tqdm, trange

from models.features import MultimodalFeatures
from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big


def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train_CFM(args):

    set_seeds()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project = 'crossmodal-feature-mappings',
        name = model_name
    )

    # Dataloader.
    train_loader = get_data_loader("train", class_name = args.class_name, img_size = 224, dataset_path = args.dataset_path, batch_size = args.batch_size, shuffle = True)
    
    # Feature extractors.
    feature_extractor = MultimodalFeatures()

    # Model instantiation. 
    CFM_2Dto3D = FeatureProjectionMLP(in_features = 768, out_features = 1152)
    CFM_3Dto2D = FeatureProjectionMLP(in_features = 1152, out_features = 768)

    optimizer = torch.optim.Adam(params = chain(CFM_2Dto3D.parameters(), CFM_3Dto2D.parameters()))

    CFM_2Dto3D.to(device), CFM_3Dto2D.to(device)

    metric = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for epoch in trange(args.epochs_no, desc = f'Training Feature Transfer Net.'):

        epoch_cos_sim_3Dto2D, epoch_cos_sim_2Dto3D = [], []

        # ------------ [Trainig Loop] ------------ #
        # * Return (rgb_img, organized_pc, depth_map_3channel), globl_label
        for (rgb, pc, _), _ in tqdm(train_loader, desc = f'Extracting feature from class: {args.class_name}.'):
            rgb, pc = rgb.to(device), pc.to(device)

            # Make CFMs trainable.
            CFM_2Dto3D.train(), CFM_3Dto2D.train()

            if args.batch_size == 1:
                rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb, pc)
            else:
                rgb_patches = []
                xyz_patches = []

                for i in range(rgb.shape[0]):
                    rgb_patch, xyz_patch = feature_extractor.get_features_maps(rgb[i].unsqueeze(dim=0), pc[i].unsqueeze(dim=0))

                    rgb_patches.append(rgb_patch)
                    xyz_patches.append(xyz_patch)

                rgb_patch = torch.stack(rgb_patches, dim=0)
                xyz_patch = torch.stack(xyz_patches, dim=0)
            
            # Predictions.
            rgb_feat_pred = CFM_3Dto2D(xyz_patch)
            xyz_feat_pred = CFM_2Dto3D(rgb_patch)

            # Losses.
            xyz_mask = (xyz_patch.sum(axis = -1)  == 0) # Mask only the feature vectors that are 0 everywhere.
            
            loss_2Dto3D = 1 - metric(xyz_feat_pred[~xyz_mask], xyz_patch[~xyz_mask]).mean()
            loss_3Dto2D = 1 - metric(rgb_feat_pred[~xyz_mask], rgb_patch[~xyz_mask]).mean()
            
            cos_sim_3Dto2D, cos_sim_2Dto3D = 1 - loss_3Dto2D.cpu(), 1 - loss_2Dto3D.cpu()

            epoch_cos_sim_3Dto2D.append(cos_sim_3Dto2D), epoch_cos_sim_2Dto3D.append(cos_sim_2Dto3D)

            # Logging.
            wandb.log({
                "train/loss_3Dto2D" : loss_3Dto2D,
                "train/loss_2Dto3D" : loss_2Dto3D,
                "train/cosine_similarity_3Dto2D" : cos_sim_3Dto2D,
                "train/cosine_similarity_2Dto3D" : cos_sim_2Dto3D,
                })

            if torch.isnan(loss_3Dto2D) or torch.isinf(loss_3Dto2D) or torch.isnan(loss_2Dto3D) or torch.isinf(loss_2Dto3D):
                exit()

            # Optimization.
            if not torch.isnan(loss_3Dto2D) and not torch.isinf(loss_3Dto2D) and not torch.isnan(loss_2Dto3D) and not torch.isinf(loss_2Dto3D):
                
                optimizer.zero_grad()

                loss_3Dto2D.backward(), loss_2Dto3D.backward()

                optimizer.step()

        # Global logging.
        wandb.log({
            "global_train/cos_sim_3Dto2D" : torch.Tensor(epoch_cos_sim_3Dto2D, device = 'cpu').mean(),
            "global_train/cos_sim_2Dto3D" : torch.Tensor(epoch_cos_sim_2Dto3D, device = 'cpu').mean()
            })

    # Model saving.
    directory = f'{args.checkpoint_savepath}/{args.class_name}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(CFM_2Dto3D.state_dict(), os.path.join(directory, 'CFM_2Dto3D_' + model_name + '.pth'))
    torch.save(CFM_3Dto2D.state_dict(), os.path.join(directory, 'CFM_3Dto2D_' + model_name + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Crossmodal Feature Networks (CFMs) on a dataset.')

    parser.add_argument('--dataset_path', default = './datasets/mvtec3d', type = str, 
                        help = 'Dataset path.')

    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_CFM_mvtec', type = str, 
                        help = 'Where to save the model checkpoints.')
    
    parser.add_argument('--class_name', default = None, type = str, choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire",
                                                                               'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear', 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train the CFMs.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    
    args = parser.parse_args()
    train_CFM(args)