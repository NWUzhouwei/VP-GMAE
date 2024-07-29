import torch
import torch.nn as nn

import extensions.func.functional as F

__all__ = ['M2Voxelization']

class Swish(nn.Module):
	def __init__(self,inplace = True):
		super().__init__()
		self.inplace = inplace
  
	def forward(self, x):
		if self.inplace:
			x.mul_(torch.sigmoid(x))
			return x
		else:
			return x * torch.sigmoid(x)

class M2Voxelization(nn.Module):
    def __init__(self, resolution, dim = 240, normalize=True, eps=0):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
            coord: [B, 3, N]
        """
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Conv3d(dim * 2, dim, 1),
            nn.BatchNorm3d(dim),
            Swish(inplace=True)
        )

    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        avg_fea = F.avg_voxelize(features, vox_coords, self.r)
        max_fea = F.max_voxelize(features, vox_coords, self.r)
        fea = torch.cat([max_fea, avg_fea], 1) # [b,2c,r,r,r]
        fea = self.mlp.to('cuda')(fea)
        return fea, norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
