#include <stdio.h>
#include <stdlib.h>
#include<math.h>

#include "../cuda_utils.cuh"

/*
  Function: get how many points in each voxel grid
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    r3     : s, voxel cube size = r ** 3
    voxcoords : coords of each point, IntTensor[b, 3, n]
    ind    : voxel index of each point, IntTensor[b, n]
    cnt    : #points in each voxel index, IntTensor[b, s]
*/
__global__ void grid_stats_kernel(int b, int n, int r, int r2, int r3,
                                  const int *__restrict__ voxcoords,
                                  int *__restrict__ ind,
                                  int *__restrict__ cnt) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  voxcoords += batch_index * n * 3;
  ind += batch_index * n;
  cnt += batch_index * r3;

  for (int i = index; i < n; i += stride) {
    // if (ind[i] == -1)
    //   continue;
    ind[i] = voxcoords[i] * r2 + voxcoords[i + n] * r + voxcoords[i + n + n];
    atomicAdd(cnt + ind[i], 1);
  }
}

/*
  Function: get center point of each voxel grid
  Args:
    b   : batch size
    c   : #channels =3
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    cnt : #points in each voxel index, IntTensor[b, s]
    coords: point coords, FloatTensor[b, 3, n]
    center : center point of each voxel, FloatTensor[b, 3, s]
*/
__global__ void center_stats_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const int *__restrict__ cnt,
                                    const float *__restrict__ coords,
                                    float *__restrict__ center) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  coords += batch_index * c * n;
  center += batch_index * c * s;
  cnt += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    // if (pos == -1)
    //   continue;
    int cur_cnt = cnt[pos];
    if (cur_cnt > 0) {
      float div_cur_cnt = 1.0 / static_cast<float>(cur_cnt);
      for (int j = 0; j < c; j++) {
        atomicAdd(center + j * s + pos, coords[j * n + i] * div_cur_cnt);
      }
    }
  }
}

/*
  Function: get weight of each point
  Args:
    b   : batch size
    c   : #channels =3
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    coords: point coords, FloatTensor[b, 3, n]
    center : center point of each voxel, FloatTensor[b, 3, s]
    weight: point weight, FloatTensor[b, n]
*/
__global__ void weight_stats_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const float *__restrict__ coords,
                                    const float *__restrict__ center,
                                    float *__restrict__ weight) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  coords += batch_index * c * n;
  center += batch_index * c * s;
  weight += batch_index * n;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i]; 
    float d2 = 0;
    for (int j = 0; j < c; j++) {
      d2 += (coords[j*n+i]-center[j*s+pos])*(coords[j*n+i]-center[j*s+pos]);
    }
    atomicAdd(weight + i, exp(-d2/0.1)); // sigma2=5
  }
}

/*
  Function: get weight sum of each voxel
  Args:
    b   : batch size
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    weight: point weight, FloatTensor[b, n]
    sum_weight: point weight, FloatTensor[b, s]
*/
__global__ void sum_weight_stats_kernel(int b, int n, int s,
                                    const int *__restrict__ ind,
                                    const float *__restrict__ weight,
                                    float *__restrict__ sum_weight) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  weight += batch_index * n;
  sum_weight += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i]; 
    atomicAdd(sum_weight + pos, weight[i]);
  }
}

/*
  Function: guassian voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    feat: features, FloatTensor[b, c, n]
    weight: point weight, FloatTensor[b, n]
    sum_weight: point weight, FloatTensor[b, s]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void gs_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const float *__restrict__ feat,
                                    const float *__restrict__ weight,
                                    const float *__restrict__ sum_weight,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  weight += batch_index * n;
  sum_weight += batch_index * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i]; 
    for (int j = 0; j < c; j++) {
      atomicAdd(out + j * s + pos, feat[j * n + i] * weight[i] / sum_weight[pos]);
    }
  }
}

/*
  Function: guassian voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    weight: point weight, FloatTensor[b, n]
    sum_weight: point weight, FloatTensor[b, s]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void gs_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ ind,
                                         const float *__restrict__ weight,
                                         const float *__restrict__ sum_weight,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  weight += batch_index * n;
  sum_weight += batch_index * r3;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    for (int j = 0; j < c; j++) {
      atomicAdd(grad_x + j * n + i, grad_y[j * r3 + pos] * weight[i] / sum_weight[pos]);
    }
  }
}

void gs_voxelize(int b, int c, int n, int r, int r2, int r3, const int *voxcoords, const float *coords,
                  const float *feat, int *ind, int *cnt, float *center, float *weight, float *sum_weight, float *out) {
  grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, r3, voxcoords, ind, cnt);
  center_stats_kernel<<<b, optimal_num_threads(n)>>>(b, 3, n, r3, ind, cnt, coords, center);
  weight_stats_kernel<<<b, optimal_num_threads(n)>>>(b, 3, n, r3, ind, coords, center, weight);
  sum_weight_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r3, ind, weight, sum_weight);
  gs_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, ind, feat, weight, sum_weight, out);
  CUDA_CHECK_ERRORS();
}

void gs_voxelize_grad(int b, int c, int n, int s, const int *ind,
                       const float *weight, const float *sum_weight, 
                       const float *grad_y, float *grad_x) {
  gs_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, ind, weight, sum_weight,
                                                          grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
