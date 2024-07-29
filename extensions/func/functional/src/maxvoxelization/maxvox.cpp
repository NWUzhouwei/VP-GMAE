#include "maxvox.hpp"
#include "maxvox.cuh"

#include "../utils.hpp"

/*
  Function: max pool voxelization (forward)
  Args:
    features: features, FloatTensor[b, c, n]
    coords  : coords of each point, IntTensor[b, 3, n]
    resolution : voxel resolution
  Return:
    out : outputs, FloatTensor[b, c, s], s = r ** 3
    ind : voxel index of each point, IntTensor[b, n]
*/
std::vector<at::Tensor> max_voxelize_forward(const at::Tensor features,
                                             const at::Tensor coords,
                                             const int resolution) {
  CHECK_CUDA(features);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int r = resolution;
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor ind = torch::zeros(
      {b, n}, at::device(features.device()).dtype(at::ScalarType::Int));
  at::Tensor midx = torch::zeros(
      {b, c, r3}, at::device(features.device()).dtype(at::ScalarType::Int));
  at::Tensor out = torch::zeros(
      {b, c, r3}, at::device(features.device()).dtype(at::ScalarType::Float));
  max_voxelize(b, c, n, r, r2, r3, coords.data_ptr<int>(),
               features.data_ptr<float>(), ind.data_ptr<int>(),
               midx.data_ptr<int>(), out.data_ptr<float>());
  return {out, ind, midx};
}

/*
  Function: max pool voxelization (backward)
  Args:
    grad_y : grad outputs, FloatTensor[b, c, s]
    indices: voxel index of each point, IntTensor[b, n]
  Return:
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
at::Tensor max_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor midex) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CUDA(midex);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(midex);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_INT(midex);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int s = grad_y.size(2);
  int n = indices.size(1);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  max_voxelize_grad(b, c, n, s, indices.data_ptr<int>(), midex.data_ptr<int>(),
                    grad_y.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}
