import torch
import torch.nn as nn

import extensions.func.functional as F
import numpy as np
import random

__all__ = ['MRVoxelization']

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

class MRVoxelization(nn.Module):
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
            nn.Conv1d(dim * 2, dim, 1),
            nn.BatchNorm1d(dim),
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
        vox_coords = torch.round(norm_coords).to(torch.int32) # [b,3,n] 素体网格的坐标
        
        b, c, n = features.shape
        # [b,n]
        pos_vox = vox_coords[:, 0, :] + vox_coords[:, 1, :] * self.r + vox_coords[:, 2, :] * self.r * self.r
        fea = torch.zeros([b, 2 * c, self.r*self.r*self.r], dtype=torch.float) # [b,2c,r^3]
        sorted_pos, index = torch.sort(pos_vox, dim = -1, descending = True) # [b, n]从大到小排序
        for i in range(b):
            beging = 0
            for j in range(n - 1):
                if sorted_pos[i, j] == sorted_pos[i, j + 1]:
                    continue
                else:
                    fea_idx = index[i, beging : j + 1]
                    fea_part = features[i, :, fea_idx] # [c,m] numpy
                    fea_part, _ = torch.sort(fea_part, dim = -1) # 从小到大排序 [c,m]
                    m = fea_part.shape[-1]
                    max_fea = fea_part[:, -1] # [c]
                    rand_fea = torch.zeros(c).cuda()
                    for k in range(c):
                        if m > 2:
                            rand_fea[k] = fea_part[k, random.randint(0, m - 2)]
                        else:
                            rand_fea[k] = fea_part[k, 0]
                    fea[i, :, sorted_pos[i, j]] = torch.cat([max_fea,rand_fea], 0)
        fea = self.mlp.to('cuda')(fea.cuda())
        return fea.view(b, c, self.r, self.r, self.r), norm_coords
        
    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')