#ifndef _MAXVOX_HPP
#define _MAXVOX_HPP

#include <torch/torch.h>
#include <vector>

std::vector<at::Tensor> max_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution);

at::Tensor max_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor midex);

#endif
