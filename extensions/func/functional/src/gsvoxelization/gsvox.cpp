#include "gsvox.hpp"
#include "gsvox.cuh"

#include "../utils.hpp"

/*
  Function: guassian voxelization (forward)
  Args:
    features: features, FloatTensor[b, c, n]
    voxcoords : coords of each point, IntTensor[b, 3, n]
    coords: point coords, FloatTensor[b, 3, n]
    resolution : voxel resolution
  Return:
    out : outputs, FloatTensor[b, c, s], s = r ** 3
    ind : voxel index of each point, IntTensor[b, n]
    weight: point weight, FloatTensor[b, n]
    sum_weight: point weight, FloatTensor[b, s]
*/
std::vector<at::Tensor> gs_voxelize_forward(const at::Tensor features,
                                             const at::Tensor voxcoords,
                                             const at::Tensor coords,
                                             const int resolution) {
  CHECK_CUDA(features);
  CHECK_CUDA(voxcoords);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(voxcoords);
  CHECK_CONTIGUOUS(coords);
  CHECK_IS_FLOAT(features);
  CHECK_IS_INT(voxcoords);
  CHECK_IS_FLOAT(coords);

  int b = features.size(0);
  int c = features.size(1);
  int n = features.size(2);
  int r = resolution;
  int r2 = r * r;
  int r3 = r2 * r;
  at::Tensor ind = torch::zeros(
      {b, n}, at::device(features.device()).dtype(at::ScalarType::Int));
  at::Tensor out = torch::zeros(
      {b, c, r3}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor cnt = torch::zeros(
      {b, r3}, at::device(features.device()).dtype(at::ScalarType::Int));

  at::Tensor center = torch::zeros(
      {b, 3, r3}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor weight = torch::zeros(
      {b, n}, at::device(features.device()).dtype(at::ScalarType::Float));
  at::Tensor sum_weight = torch::zeros(
      {b, r3}, at::device(features.device()).dtype(at::ScalarType::Float));

  gs_voxelize(b, c, n, r, r2, r3, voxcoords.data_ptr<int>(), coords.data_ptr<float>(),
               features.data_ptr<float>(), ind.data_ptr<int>(), cnt.data_ptr<int>(),
               center.data_ptr<float>(), weight.data_ptr<float>(),
               sum_weight.data_ptr<float>(), out.data_ptr<float>());
  return {out, ind, weight, sum_weight};
}

/*
  Function: guassian voxelization (backward)
  Args:
    grad_y : grad outputs, FloatTensor[b, c, s]
    indices: voxel index of each point, IntTensor[b, n]
    weight: point weight, FloatTensor[b, n]
    sum_weight: point weight, FloatTensor[b, s]
  Return:
    grad_x : grad inputs, FloatTensor[b, c, n]
*/
at::Tensor gs_voxelize_backward(const at::Tensor grad_y,
                                 const at::Tensor indices,
                                 const at::Tensor weight,
                                 const at::Tensor sum_weight) {
  CHECK_CUDA(grad_y);
  CHECK_CUDA(indices);
  CHECK_CUDA(weight);
  CHECK_CUDA(sum_weight);
  CHECK_CONTIGUOUS(grad_y);
  CHECK_CONTIGUOUS(indices);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(sum_weight);
  CHECK_IS_FLOAT(grad_y);
  CHECK_IS_INT(indices);
  CHECK_IS_FLOAT(weight);
  CHECK_IS_FLOAT(sum_weight);

  int b = grad_y.size(0);
  int c = grad_y.size(1);
  int s = grad_y.size(2);
  int n = indices.size(1);
  at::Tensor grad_x = torch::zeros(
      {b, c, n}, at::device(grad_y.device()).dtype(at::ScalarType::Float));
  gs_voxelize_grad(b, c, n, s, indices.data_ptr<int>(), weight.data_ptr<float>(),
                    sum_weight.data_ptr<float>(), grad_y.data_ptr<float>(), grad_x.data_ptr<float>());
  return grad_x;
}
