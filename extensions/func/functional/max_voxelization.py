from torch.autograd import Function

from extensions.func.functional.backend import _backend

__all__ = ['max_voxelize']


class MAXVoxelize(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, midx = _backend.max_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, midx)
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
        indices, midex = ctx.saved_tensors
        grad_features = _backend.max_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, midex)
        return grad_features, None, None


max_voxelize = MAXVoxelize.apply
