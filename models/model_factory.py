# Author: Jacek Komorowski
# Warsaw University of Technology
import MinkowskiEngine as ME

from torchpack.utils.config import configs

import models.minkloc as minkloc
from models.PointNetVlad import PointNetVlad
from models.LOGG3D import *
from models.PatchAugNet import *
from models.QFormer import Promptor
from misc.utils import sparcify_and_collate_list, freeze_model


class LifeLongModel(torch.nn.Module):
    def __init__(self, use_prompt=False, use_scene_id=False) -> None:
        super().__init__()
        # backbone
        if 'MinkFPN' in configs.model.name:
            self.backbone = minkloc.MinkLoc(
                configs.model.name, 
                in_channels=1,
                feature_size=configs.model.feature_size,
                output_dim=configs.model.output_dim, 
                planes=configs.model.planes,
                layers=configs.model.layers, 
                num_top_down=configs.model.num_top_down,
                conv0_kernel_size=configs.model.conv0_kernel_size)
            self.prompt_dim = self.backbone.in_channels
        elif configs.model.name == 'PointNetVlad':
            self.backbone = PointNetVlad(
                num_points = configs.data.num_points,
                global_feat = True,
                feature_transform = True,
                max_pool = False,
                output_dim = configs.model.output_dim)
            self.prompt_dim = 3
        elif configs.model.name == 'logg3d':
            self.backbone = LOGG3D(output_dim=256)
            self.prompt_dim = 3
        elif configs.model.name == 'PatchAugNet':
            self.backbone = PatchAugNet()
            self.prompt_dim = 3
        else:
            raise NotImplementedError('Model not implemented: {}'.format(configs.model.name))
        
        # Prompt
        self.use_prompt = use_prompt
        self.promptor = Promptor(self.prompt_dim, use_scene_id)
    
    def prepare_before_train(self, stage=-1):
        if stage == 2:  # train modules except qformer
            state = True
        elif stage == 1:  # train qformer only
            state = False
        else:
            return
        
        for param in self.promptor.parameters():
            param.requires_grad = not state
        for param in self.backbone.parameters():
            param.requires_grad = state
    
    def forward(self, batch):
        # prompt embedding
        embedding = self.promptor(batch) if self.use_prompt else None  # B x N x prompt_dim or None, B x num_select
        
        if 'MinkFPN' in configs.model.name:
            if embedding is not None:
                coords_batch, feats_batch = [], []
                for e,f in zip(batch['cloud'], embedding):
                    coords, feats = ME.utils.sparse_quantize(coordinates=e, features=f, quantization_size=configs.model.mink_quantization_size)
                    coords_batch.append(coords)
                    feats_batch.append(feats)
                batch['coords'] = ME.utils.batched_coordinates(coords_batch)
                batch['features'] = torch.cat(feats_batch, 0).float() + torch.ones((batch['coords'].shape[0], self.prompt_dim), dtype=torch.float32).cuda()
            else:
                coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=configs.model.mink_quantization_size)
                          for e in batch['cloud']]
                batch['coords'] = ME.utils.batched_coordinates(coords)
                batch['features'] = torch.ones((batch['coords'].shape[0], self.prompt_dim), dtype=torch.float32).cuda()
        elif 'logg3d' in configs.model.name:
            voxel_size = configs.model.mink_quantization_size
            if embedding is None:
                list_pc, list_feat = [], None
                for e in batch['cloud']:
                    list_pc.append(e.detach().cpu())
                batch['coords'] = sparcify_and_collate_list(list_pc, voxel_size, list_feat)
            else:
                list_pc, list_feat = [], []
                for e,f in zip(batch['cloud'], embedding):
                    list_pc.append(e.detach().cpu())
                    list_feat.append(f.detach().cpu().numpy())
                batch['coords'] = sparcify_and_collate_list(list_pc, voxel_size, list_feat)
        else:
            batch['features'] = embedding
        x = self.backbone(batch)  # B x 256
        projector = None
        return x, projector


def model_factory(ckpt = None, use_prompt=False, use_scene_id=False, device = 'cuda'):
    model = LifeLongModel(use_prompt, use_scene_id)
    if ckpt != None:
        model.load_state_dict(ckpt)
    model = model.to(device)

    return model


def copy_frozen_model(model):
    model_frozen = model_factory(model.state_dict())
    freeze_model(model_frozen)
    return model_frozen
