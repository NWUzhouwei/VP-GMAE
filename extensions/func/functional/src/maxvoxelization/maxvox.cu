#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.cuh"

/*
  float atomicMax

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}
*/

/*
  Function: get voxel index of each point
  Args:
    b      : batch size
    n      : number of points
    r      : voxel resolution
    r2     : = r * r
    coords : coords of each point, IntTensor[b, 3, n]
    ind    : voxel index of each point, IntTensor[b, n]
*/
__global__ void grid_stats_kernel(int b, int n, int r, int r2,
                                  const int *__restrict__ coords,
                                  int *__restrict__ ind) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  coords += batch_index * n * 3;
  ind += batch_index * n;

  for (int i = index; i < n; i += stride) {
    ind[i] = coords[i] * r2 + coords[i + n] * r + coords[i + n + n];
  }
}

/*
  Function: max pool voxelization (forward)
  Args:
    b   : batch size
    c   : #channels
    n   : number of points
    s   : voxel cube size = voxel resolution ** 3
    ind : voxel index of each point, IntTensor[b, n]
    feat: features, FloatTensor[b, c, n]
    out : outputs, FloatTensor[b, c, s]
*/
__global__ void max_voxelize_kernel(int b, int c, int n, int s,
                                    const int *__restrict__ ind,
                                    const float *__restrict__ feat,
                                    int *__restrict__ midx,
                                    float *__restrict__ out) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  feat += batch_index * c * n;
  out += batch_index * c * s;
  midx += batch_index * c * s;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i]; // voxel index of point i 
    for (int j = 0; j < c; j++) {
      if(out[j * s + pos] < feat[j * n + i]) {
        out[j * s + pos] = feat[j * n + i];
        midx[j * s + pos] = j * n + i;
      }
    }
  }
}

/*
  Function: max pool voxelization (backward)
  Args:
    b      : batch size
    c      : #channels
    n      : number of points
    r3     : voxel cube size = voxel resolution ** 3
    ind    : voxel index of each point, IntTensor[b, n]
    grad_y : grad outputs, FloatTensor[b, c, s]
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
__global__ void max_voxelize_grad_kernel(int b, int c, int n, int r3,
                                         const int *__restrict__ ind,
                                         const int *__restrict__ midx,
                                         const float *__restrict__ grad_y,
                                         float *__restrict__ grad_x) {
  int batch_index = blockIdx.x;
  int stride = blockDim.x;
  int index = threadIdx.x;
  ind += batch_index * n;
  midx += batch_index * c * r3;
  grad_x += batch_index * c * n;
  grad_y += batch_index * c * r3;
  for (int i = index; i < n; i += stride) {
    int pos = ind[i];
    for (int j = 0; j < c; j++) {
      grad_x[midx[j * r3 + pos]] = grad_y[j * r3 + pos];
    }
  }
}

void max_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *midx, float *out) {
  grid_stats_kernel<<<b, optimal_num_threads(n)>>>(b, n, r, r2, coords, ind);
  max_voxelize_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, r3, ind, feat, midx, out);
  CUDA_CHECK_ERRORS();
}

void max_voxelize_grad(int b, int c, int n, int s, const int *ind, const int *midx,
                       const float *grad_y, float *grad_x) {
  max_voxelize_grad_kernel<<<b, optimal_num_threads(n)>>>(b, c, n, s, ind, midx,
                                                          grad_y, grad_x);
  CUDA_CHECK_ERRORS();
}
