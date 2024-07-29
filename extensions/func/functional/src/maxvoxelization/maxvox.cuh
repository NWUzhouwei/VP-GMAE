#ifndef _MAXVOX_CUH
#define _MAXVOX_CUH

// CUDA function declarations
void max_voxelize(int b, int c, int n, int r, int r2, int r3, const int *coords,
                  const float *feat, int *ind, int *midx, float *out);
void max_voxelize_grad(int b, int c, int n, int s, const int *idx, const int *midx,
                       const float *grad_y, float *grad_x);

#endif
