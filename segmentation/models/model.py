import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from logger import get_missing_parameters_message, get_unexpected_parameters_message
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from extensions.func import Voxelization, MaxVoxelization, M2Voxelization, MRVoxelization
from extensions.func.functional import grouping, trilinear_devoxelize
from models.resolution_transform import VoxelTrans as VoxelTransform


def knn(xm, xn, k):
    '''
    xm: [b, M, C]
    xn: [b, N, C]
    k: neighbor

    return: idx: [b, M, k]
    '''
    # [b,m,1,c] [b,1,n,c]->[b,m,n,c]->[b,m,n]
    distance = torch.sum((xm[:, :, None] - xn[:, None]) ** 2, dim=-1) # [b,m,n]
    idx = distance.argsort()[:, :, :k] # [b,m,k]
    return idx # [b,m,k]

# Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = Swish()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, N, C = x.shape
        qkv = self.qkv(x) # [B,S,N,3C]
        qkv = qkv.reshape(B,S,N,3,self.num_heads, C//self.num_heads).permute(3,0,4,1,2,5)
        qkv = qkv.reshape(3,B,self.num_heads,S,-1) # [3,b,6,s,nc/6]
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2] # [b,6,s,nc/6]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, S, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=1., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x): # [b,s,n,c]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=384, depth=12, num_heads=6, mlp_ratio=1., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=1., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x)
        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x

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

class Dgcnn(nn.Module):
    def __init__(self, encoder_channel, k):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.k = k
        self.mlp = Mlp(in_features = 6, hidden_features=self.encoder_channel//2, out_features=self.encoder_channel)

    def forward(self, point):
        '''
            point : B N 3      B G N 3
            -----------------
            return : B N C
        '''
        idx = knn(point.contiguous(), point.contiguous(), self.k) # [b,n,k]
        neigh = grouping(point.transpose(2, 1), idx.int()).permute(0,2,3,1) # [b,n,k,3]
        neigh = neigh - point.unsqueeze(2) # [b,n,k,3]
        f = torch.cat([neigh, point.unsqueeze(2).repeat(1,1,self.k,1)], -1) # [b,n,k,6]
        f = self.mlp(f) # [b,n,k,c]
        f = f.max(-2)[0] # [b,n,c]

        return f
    
class MiniPointNet(nn.Module):  # Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            Swish(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            Swish(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point):
        '''
            point : B N 3      B G N 3
            -----------------
            return : B N C
        '''
        _, n, _ = point.shape
        feature = self.first_conv(point.transpose(2, 1)) # [b,256,n]
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1) # [b,512,n]
        feature = self.second_conv(feature) # [b,c,n]
        return feature.transpose(2, 1)
    
class VoxelMaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.depth = config.transformer_config.depth # 12
        self.r_list = config.transformer_config.resolution # voxel resolution
        self.length = len(self.r_list)
        self.mask_ratio = config.transformer_config.mask_ratio # 0.5
        self.drop_path_rate = config.transformer_config.drop_path_rate # 0.1
        self.num_heads = config.transformer_config.num_heads # 6
        self.trans_dim = config.dims # 384
        self.voxelize_type = config.voxelize_type
        # define the encoder
        # bridge inpute and transformer
        assert self.voxelize_type in ['avg', 'max', 'm2', 'mr']
        if self.voxelize_type == 'avg':
            self.voxelizations = [Voxelization(resolution = self.r_list[i]) for i in range(self.length)]
        elif self.voxelize_type == 'max':
            self.voxelizations = [MaxVoxelization(resolution = self.r_list[i]) for i in range(self.length)]
        elif self.voxelize_type == 'm2':
            self.voxelizations = [M2Voxelization(resolution = self.r_list[i]) for i in range(self.length)]
        elif self.voxelize_type == 'mr':
            self.voxelizations = [MRVoxelization(resolution = self.r_list[i]) for i in range(self.length)]
            
        self.c = nn.Sequential(
            nn.Conv2d(self.trans_dim, self.trans_dim, kernel_size = 1),
            nn.BatchNorm2d(self.trans_dim),
            Swish()
        )
        
        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        
        # layer norm
        self.norm = nn.LayerNorm([self.r_list[-1]**3, self.trans_dim])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def _mask_generate(self, feature, skip = False):
        '''
            feature : B S NC
            --------------
            mask : B S (bool) mask set 1; unmask set 0
        '''
        B, S, _, _ = feature.shape
        # skip the mask
        if skip:
            return torch.zeros(B, S).bool()
        # mask a continuous part
        self.num_mask = S/2
        mask_list = []
        for i in range(S):
            if (i+1)%2==1: # mask
                mask_list.append(torch.ones(B).bool()) # B
            else: # unmask
                mask_list.append(torch.zeros(B).bool()) # B
        mask = torch.stack(mask_list, dim = 1) #.view(B, S, 1).repeat(1, 1, NC)
            
        return mask# .to(feature.device) # [b,s #,nc]
    
    def _rand_mask_generate(self, feature, skip = False):
        '''
            feature : B S N C
            --------------
            mask : B S (bool) mask set 1; unmask set 0
        '''
        B, S, _, _ = feature.shape
        # skip the mask
        if skip:
            return torch.zeros(B, S).bool()
        # mask a continuous part
        self.num_mask = int(self.mask_ratio * S)
        overall_mask = np.zeros([B, S])
        for i in range(B):
            mask = np.hstack([
                np.zeros(S-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        
        return overall_mask.to(feature.device) # [b,s]

    def forward(self, xyz, feature, classify=False): # [b,n,c]
        b, _, c = feature.shape
        v_fea_list = []
        for i in range(self.length):
            # [b,c,r,r,r]
            voxel_feature, _ = self.voxelizations[i](feature.permute(0,2,1), xyz.permute(0,2,1))
            v_fea_list.append(voxel_feature)
        
        fea_list = VoxelTransform(v_fea_list, r_list = self.r_list) # 4*[b,n,c] n=r**3
        fea_token = torch.stack(fea_list, dim=1) # [b,4,n,c]
        # [b,4,n,c]
        fea_token = self.c(fea_token.permute(0,3,1,2)).permute(0,2,3,1)
        # generate mask
        mask = self._rand_mask_generate(fea_token) # [b,s]
        x_vis = fea_token[~mask].reshape(b, self.length - self.num_mask, -1, c) # [b,2,nc]
        
        # transformer
        x_vis = self.blocks(x_vis) # [b,s/2,n,c]
        x_vis = self.norm(x_vis)

        return fea_token, x_vis, mask

class get_model(nn.Module):
    def __init__(self, cls_dim):
        super().__init__()

        self.trans_dim = 240 # 384
        self.depth = 12
        self.drop_path_rate = 0.1
        self.cls_dim = cls_dim
        self.num_heads = 6
        self.r_list = [2, 3, 5, 7]
        self.length = len(self.r_list)
        self.voxelize_type = "avg" # ['avg', 'max', 'm2', 'mr']
        self.embedding = "dgcnn" # ['pointnet', 'dgcnn']
        self.dgcnn_k = 32
        
        if self.embedding == 'pointnet':
            self.input_embedding = MiniPointNet(self.trans_dim)
        elif self.embedding == 'dgcnn':
            self.input_embedding = Dgcnn(self.trans_dim, self.dgcnn_k)
        
        if self.voxelize_type == 'avg':
            self.voxelizations = [Voxelization(resolution = self.r_list[i]) for i in range(self.length)]
        elif self.voxelize_type == 'max':
            self.voxelizations = [MaxVoxelization(resolution = self.r_list[i]) for i in range(self.length)]
        elif self.voxelize_type == 'm2':
            self.voxelizations = [M2Voxelization(resolution = self.r_list[i]) for i in range(self.length)]
        elif self.voxelize_type == 'mr':
            self.voxelizations = [MRVoxelization(resolution = self.r_list[i]) for i in range(self.length)]
        
        self.c = nn.Sequential(
            nn.Conv2d(self.trans_dim, self.trans_dim, kernel_size = 1),
            nn.BatchNorm2d(self.trans_dim),
            Swish()
        )
        
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )
        
        self.norm = nn.LayerNorm([self.r_list[-1]**3, self.trans_dim])
        
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(0.2))

        self.convs1 = nn.Conv1d(9 * self.trans_dim + 64, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.cls_dim, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100
    
    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('encoder'):
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts, cls_label): # pts:[b,n,c]  cls_label:[b,16]
        B, N, _ = pts.shape
        
        feature = self.input_embedding(pts) # [b,n,c]
        v_fea_list = []
        for i in range(self.length):
            # [b,c,r,r,r]
            voxel_feature, norm_coords = self.voxelizations[i](feature.permute(0,2,1), pts.permute(0,2,1))
            v_fea_list.append(voxel_feature)
        
        fea_list = VoxelTransform(v_fea_list, r_list = self.r_list) # 4*[b,n,c]
        fea_token = torch.stack(fea_list, dim=1) # [b,4,n,c]
        
        # [b,s,n,c]
        fea_token = self.c(fea_token.permute(0,3,1,2)).permute(0,2,3,1)
        feature_list = self.blocks(fea_token) # [b,s,n,c]
        feature_list = [self.norm(x).contiguous() for x in feature_list]
        x = torch.cat((feature_list[0],feature_list[1],feature_list[2]), dim=-1) # [b,s,n,3c]
        x_max = torch.max(x.reshape(B, -1, x.shape[-1]), 1)[0] # [b,3c]
        x_avg = torch.mean(x.reshape(B, -1, x.shape[-1]), 1) # [b,3c]
        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N) # [b,3c,N]
        x_avg_feature = x_avg.unsqueeze(-1).repeat(1, 1, N) # [b,3c,N]
        cls_label_one_hot = cls_label.view(B, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N) # [b,64,N]
        x_global_feature = torch.cat((x_max_feature, x_avg_feature, cls_label_feature), 1) # [b,6c+64,N]

        R = self.r_list[-1]
        x_v = torch.max(x, 1)[0].reshape(B, R, R, R, -1).permute(0, 4, 1, 2, 3) # [b,r,r,r,3c]
        x_n = trilinear_devoxelize(x_v, norm_coords, R) # [b,3c,N]
        x = torch.cat((x_n, x_global_feature), 1) # [b,9c+64,N]
        x = self.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss