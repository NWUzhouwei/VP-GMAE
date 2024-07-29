import torch
import numpy as np

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, M, [K]]
    Return:
        new_points:, indexed points data, [B, M, [K], C]
    """
    raw_size = idx.size() # B M K
    idx = idx.reshape(raw_size[0], -1) # B MK
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def get_flag(r_list):
    rm = r_list[-1] # rmax
    flag = np.empty([len(r_list) - 1, rm], dtype=int)
    for i in range(len(r_list) - 1):
        r = r_list[i] # r
        oneflag = np.array([])
        a = rm % r
        if a == 0:
            for j in range(r):
                for _ in range(rm // r):
                    oneflag = np.append(oneflag, j)
        elif r % a == 0:
            for j in range(r):
                if j % (r/a) == 0:
                    for _ in range(rm // r + 1):
                        oneflag = np.append(oneflag, j)
                else:
                    for _ in range(rm // r):
                        oneflag = np.append(oneflag, j)
        elif (r+1) % a == 0:
            for j in range(r):
                if j % ((r+1)/a) == 0:
                    for _ in range(rm // r + 1):
                        oneflag = np.append(oneflag, j)
                else:
                    for _ in range(rm // r):
                        oneflag = np.append(oneflag, j)
        elif r % a == 1:
            if r // a == 1:
                for j in range(r - 1):
                    for _ in range(rm // r + 1):
                        oneflag = np.append(oneflag, j)
                for _ in range(rm // r):
                    oneflag = np.append(oneflag, r - 1)
            else:
                for j in range(r):
                    if j % (r // a + 1) == 0:
                        for _ in range(rm // r + 1):
                            oneflag = oneflag = np.append(oneflag, j)
                    else:
                        for _ in range(rm // r):
                            oneflag = np.append(oneflag, j)
        else:
            for j in range(a):
                for _ in range(rm // r + 1):
                    oneflag = np.append(oneflag, j)
            for j in range(a, r):
                for _ in range(rm // r):
                    oneflag = np.append(oneflag, j)
        flag[i, :] = oneflag
    return flag

def VoxelTrans(v_fea_list, r_list):
    b, c, _, _, _ = v_fea_list[0].shape
    fe_list = []
    
    if len(r_list) == 1:
        feature = v_fea_list[-1].reshape(b, c, -1).permute(0, 2, 1) # [b, n, c]
        fe_list.append(feature)
        return fe_list
    
    flag = get_flag(r_list)
    for k in range(len(r_list)-1):
        l = []
        r = r_list[k]
        v_flage = np.array(range(r**3)).reshape(r,r,r)
        for i1 in range(r_list[-1]):
            for i2 in range(r_list[-1]):
                for i3 in range(r_list[-1]):
                    l.append(v_flage[flag[k,i1], flag[k,i2], flag[k,i3]])
        
        feature = v_fea_list[k].reshape(b, c, -1).permute(0, 2, 1) # [b, n, c]
        idx = torch.LongTensor(l).view(1, r_list[-1]**3).repeat(b, 1) # [b,n]
        fe_list.append(index_points(feature, idx.to(feature.device)))
    
    feature = v_fea_list[-1].reshape(b, c, -1).permute(0, 2, 1) # [b, n, c]
    fe_list.append(feature)
    return fe_list # 4*[b,n,c]
