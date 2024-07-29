import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils.logger import *
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from models.resolution_transform import VoxelTrans as VoxelTransform
from extensions.func import Voxelization, MaxVoxelization, M2Voxelization, MRVoxelization
from extensions.func.functional import grouping
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

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

class Attention1(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3 * 1000, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, S, N, C = x.shape
        x = x.reshape(B, S, -1) # [b, s, NC]
        qkv = self.qkv(x) # [B,S,3NC]
        qkv = qkv.reshape(B,S,3,self.num_heads, N * C//self.num_heads).permute(2,0,3,1,4)
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
        for _, block in enumerate(self.blocks):
            x = block(x)
        return x
    
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

@MODELS.register_module()
class VSSLPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dims = config.dims
        assert config.input_embedding in ['pointnet', 'dgcnn']
        if config.input_embedding == 'pointnet':
            self.input_embedding = MiniPointNet(self.dims)
        elif config.input_embedding == 'dgcnn':
            self.input_embedding = Dgcnn(self.dims, config.dgcnn_k)
        self.encoder = VoxelMaskTransformer(config)
        
        assert config.middle_mlp in ['T', 'F']
        self.midflag = config.middle_mlp
        if self.midflag == 'T':
            self.midmlp = Mlp(in_features = self.dims)
            
        self.drop_path_rate = config.transformer_config.drop_path_rate # 0.1
        self.decoder_depth = config.transformer_config.decoder_depth # 4
        self.decoder_num_heads = config.transformer_config.decoder_num_heads # 6
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.decoder = TransformerDecoder(
            embed_dim=self.dims,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        r_list = config.transformer_config.resolution
        self.length = len(r_list)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, r_list[-1]**3, self.dims))
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        self.build_loss_func(self.loss)
        
    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'l1':
            self.loss_func = torch.nn.L1Loss(reduction='mean')
        else:
            raise NotImplementedError
        
    def forward(self, pts, **kwargs):
        '''
        pts: B N 3
        '''
        feature = self.input_embedding(pts) # [b,n,c] point feature 256
        fea_token, x_vis, mask = self.encoder(pts, feature) # [b,s/2,n,c] [b,s]
        B,VIS,N,C = x_vis.shape # [b,vis,n,c]
        
        if self.midflag == 'T':
            x_vis = self.midmlp(x_vis)
        
        mask_num = self.length - VIS
        mask_token = self.mask_token.expand(B, mask_num, -1, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1) # [b,s,n,c]
        
        # decoder only mask
        x_rec = self.decoder(x_full, mask_num) # [b,s/2,n,c]
        rebuild_voxel = x_rec.reshape(B, mask_num, N, C) # [b,s/2,n,c]
        gt_voxel = fea_token[mask].reshape(B, mask_num, N, C) # [b,s/2,n,c]
        
        loss = self.loss_func(rebuild_voxel, gt_voxel)
        
        return loss
    
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.dims
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.transformer_config.num_heads
        self.r_list = config.transformer_config.resolution
        self.length = len(self.r_list)
        self.voxelize_type = config.voxelize_type
        
        assert config.input_embedding in ['pointnet', 'dgcnn']
        if config.input_embedding == 'pointnet':
            self.input_embedding = MiniPointNet(self.trans_dim)
        elif config.input_embedding == 'dgcnn':
            self.input_embedding = Dgcnn(self.trans_dim, config.dgcnn_k)
        
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
        
        self.norm = nn.LayerNorm([self.r_list[-1]**3, self.trans_dim])

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * self.length, 1024),
                nn.BatchNorm1d(1024),
                Swish(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                Swish(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                Swish(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                Swish(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

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
                if k.startswith('encoder') :
                    base_ckpt[k[len('encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
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

    def forward(self, pts):
        feature = self.input_embedding(pts) # [b,n,c]
        b, _, c = feature.shape
        v_fea_list = []
        for i in range(self.length):
            # [b,c,r,r,r]
            voxel_feature, _ = self.voxelizations[i](feature.permute(0,2,1), pts.permute(0,2,1))
            v_fea_list.append(voxel_feature)
        
        fea_list = VoxelTransform(v_fea_list, r_list = self.r_list) # 4*[b,n,c]
        fea_token = torch.stack(fea_list, dim=1) # [b,4,n,c]
        # [b,s,n,c]
        fea_token = self.c(fea_token.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.blocks(fea_token)
        x = self.norm(x) # [b,s,n,c]
        f1 = x.max(-2)[0] # [b,s,c]
        f2 = f1.reshape(b, -1) # [b, sc]
        ret = self.cls_head_finetune(f2)
        return ret