from torch.autograd import Function

from extensions.func.functional.backend import _backend

__all__ = ['gs_voxelize']


class GSVoxelize(Function):
    @staticmethod
    def forward(ctx, features, voxcoords, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param voxcoords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param coords: Coordinates of each point, FloatTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        voxcoords = voxcoords.int().contiguous()
        coords = coords.contiguous()
        b, c, _ = features.shape
        out, indices, weight, sum_weight = _backend.gs_voxelize_forward(features, voxcoords, coords, resolution)
        ctx.save_for_backward(indices, weight, sum_weight)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, weight, sum_weight = ctx.saved_tensors
        grad_features = _backend.gs_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, weight, sum_weight)
        return grad_features, None, None, None


gs_voxelize = GSVoxelize.apply
