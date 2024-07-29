#ifndef _GSVOX_HPP
#define _GSVOX_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> gs_voxelize_forward(const at::Tensor features,
                                             const at::Tensor voxcoords,
                                             const at::Tensor coords,
                                             const int resolution);

at::Tensor gs_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor weight,
                                 const at::Tensor sum_weight);

#endif
