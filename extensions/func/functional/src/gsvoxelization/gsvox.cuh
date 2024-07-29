#ifndef _GSVOX_CUH
#define _GSVOX_CUH

// CUDA function declarations
void gs_voxelize(int b, int c, int n, int r, int r2, int r3, const int *voxcoords,
                  const float *coords, const float *feat, int *ind, int *cnt,
                  float *center, float *weight, float *sum_weight, float *out);
void gs_voxelize_grad(int b, int c, int n, int s, const int *idx,
                       const float *weight, const float *sum_weight,
                       const float *grad_y, float *grad_x);

#endif
