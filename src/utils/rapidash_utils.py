import math
import numbers
import os.path as osp
import random
import time
from typing import Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch

try:
    import torch_geometric.transforms as T
    from torch_geometric.data import Data
    from torch_geometric.datasets import ShapeNet
    from torch_geometric.loader import DataLoader
    from torch_geometric.transforms import BaseTransform, LinearTransformation
    HAS_PYG = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PYG = False
    # Minimal fallback stubs so downstream imports succeed without torch_geometric.
    class BaseTransform:
        def __call__(self, data):
            return data
    class Data:
        pass


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) * 1.0 / (self.warmup + 1e-6)
        return lr_factor



if HAS_PYG:
    class RandomRotateWithNormals(BaseTransform):
        r"""Rotates node positions around a specific axis by a randomly sampled
        factor within a given interval (functional name: :obj:`random_rotate`).

        Args:
            degrees (tuple or float): Rotation interval from which the rotation
                angle is sampled. If :obj:`degrees` is a number instead of a
                tuple, the interval is given by :math:`[-\mathrm{degrees},
                \mathrm{degrees}]`.
            axis (int, optional): The rotation axis. (default: :obj:`0`)
        """
        def __init__(self, degrees: Union[Tuple[float, float], float],
                     axis: int = 0):
            if isinstance(degrees, numbers.Number):
                degrees = (-abs(degrees), abs(degrees))
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
            self.degrees = degrees
            self.axis = axis

        def __call__(self, data: Data) -> Data:
            degree = math.pi * random.uniform(*self.degrees) / 180.0
            sin, cos = math.sin(degree), math.cos(degree)

            if data.pos.size(-1) == 2:
                matrix = [[cos, sin], [-sin, cos]]
            else:
                if self.axis == 0:
                    matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
                elif self.axis == 1:
                    matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
                else:
                    matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
            return LinearTransformationWithNormals(torch.tensor(matrix))(data)

        def __repr__(self) -> str:
            return (f'{self.__class__.__name__}({self.degrees}, '
                    f'axis={self.axis})')
        
    class LinearTransformationWithNormals(BaseTransform):
        r"""Transforms node positions with a square transformation matrix computed
        offline.

        Args:
            matrix (Tensor): tensor with shape :obj:`[D, D]` where :obj:`D`
                corresponds to the dimensionality of node positions.
        """
        def __init__(self, matrix):
            assert matrix.dim() == 2, (
                'Transformation matrix should be two-dimensional.')
            assert matrix.size(0) == matrix.size(1), (
                'Transformation matrix should be square. Got [{} x {}] rectangular'
                'matrix.'.format(*matrix.size()))

            # Store the matrix as its transpose.
            # We do this to enable post-multiplication in `__call__`.
            self.matrix = matrix.t()

        def __call__(self, data):
            pos = data.pos.view(-1, 1) if data.pos.dim() == 1 else data.pos
            norm = data.x.view(-1, 1) if data.x.dim() == 1 else data.x

            assert pos.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            assert norm.size(-1) == self.matrix.size(-2), (
                'Node position matrix and transformation matrix have incompatible '
                'shape.')

            # We post-multiply the points by the transformation matrix instead of
            # pre-multiplying, because `data.pos` has shape `[N, D]`, and we want
            # to preserve this shape.
            data.pos = torch.matmul(pos, self.matrix.to(pos.dtype).to(pos.device))
            data.x = torch.matmul(norm, self.matrix.to(norm.dtype).to(norm.device))

            return data

        def __repr__(self):
            return '{}({})'.format(self.__class__.__name__, self.matrix.tolist())

if HAS_PYG:
    # Adapted from pytorch geometric, but updated the cross product as not to procude warnings
    class SamplePoints(BaseTransform):
        """Uniformly samples a fixed number of points on the mesh faces according
        to their face area.

        Args:
            num (int): The number of points to sample.
            remove_faces (bool, optional): If set to :obj:`False`, the face tensor
                will not be removed. (default: :obj:`True`)
            include_normals (bool, optional): If set to :obj:`True`, then compute
                normals for each sampled point. (default: :obj:`False`)
        """
        def __init__(
            self,
            num: int,
            remove_faces: bool = True,
            include_normals: bool = False,
        ):
            self.num = num
            self.remove_faces = remove_faces
            self.include_normals = include_normals

        def forward(self, data: Data) -> Data:
            assert data.pos is not None
            assert data.face is not None

            pos, face = data.pos, data.face
            assert pos.size(1) == 3 and face.size(0) == 3

            # Normalize positions
            pos_max = pos.abs().max()
            pos = pos / pos_max

            # Calculate face areas using linalg.cross
            vec1 = pos[face[1]] - pos[face[0]]
            vec2 = pos[face[2]] - pos[face[0]]
            area = torch.linalg.cross(vec1, vec2, dim=1)
            area = area.norm(p=2, dim=1).abs() / 2

            # Sample points based on face areas
            prob = area / area.sum()
            sample = torch.multinomial(prob, self.num, replacement=True)
            face = face[:, sample]

            # Generate random barycentric coordinates
            frac = torch.rand(self.num, 2, device=pos.device)
            mask = frac.sum(dim=-1) > 1
            frac[mask] = 1 - frac[mask]

            # Calculate vectors for point sampling
            vec1 = pos[face[1]] - pos[face[0]]
            vec2 = pos[face[2]] - pos[face[0]]

            # Compute normals if requested
            if self.include_normals:
                normals = torch.linalg.cross(vec1, vec2, dim=1)
                data.normal = torch.nn.functional.normalize(normals, p=2, dim=1)

            # Sample points using barycentric coordinates
            pos_sampled = pos[face[0]]
            pos_sampled += frac[:, :1] * vec1
            pos_sampled += frac[:, 1:] * vec2

            # Restore original scale
            pos_sampled = pos_sampled * pos_max
            data.pos = pos_sampled

            if self.remove_faces:
                data.face = None

            return data
        
    class NormalizeCoord(BaseTransform):
        """
        Normalizes the point cloud coordinates by:
        1. Centering them at the origin (zero mean)
        2. Scaling by the maximum distance from origin
        """
        def __init__(self):
            super().__init__()

        def __call__(self, data):
            # Center the points by subtracting mean
            centroid = torch.mean(data.pos, dim=0)
            data.pos = data.pos - centroid

            # Scale by maximum distance from origin
            distances = torch.sqrt(torch.sum(data.pos ** 2, dim=1))
            scale = torch.max(distances)
            data.pos = data.pos / scale

            return data

    class RandomJitter(BaseTransform):
        """Randomly jitter points by adding normal noise."""
        def __init__(self, sigma=0.01, clip=0.05):
            self.sigma = sigma
            self.clip = clip

        def __call__(self, data):
            noise = torch.clamp(
                self.sigma * torch.randn_like(data.pos), 
                min=-self.clip, 
                max=self.clip
            )
            data.pos = data.pos + noise
            return data

class RandomShift(BaseTransform):
    """Randomly shift the point cloud."""
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, data):
        shift = torch.rand(3) * 2 * self.shift_range - self.shift_range
        shift = shift.to(data.pos.device)
        data.pos = data.pos + shift
        return data
    
class RandomRotatePerturbation(BaseTransform):
    """Apply small random rotations around all axes."""
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma = angle_sigma
        self.angle_clip = angle_clip

    def __call__(self, data):
        angles = torch.clamp(
            self.angle_sigma * torch.randn(3),
            min=-self.angle_clip,
            max=self.angle_clip
        ).to(data.pos.device)

        # Create rotation matrices for each axis
        cos_x, sin_x = torch.cos(angles[0]), torch.sin(angles[0])
        cos_y, sin_y = torch.cos(angles[1]), torch.sin(angles[1])
        cos_z, sin_z = torch.cos(angles[2]), torch.sin(angles[2])

        Rx = torch.tensor([[1, 0, 0],
                          [0, cos_x, -sin_x],
                          [0, sin_x, cos_x]], device=data.pos.device)

        Ry = torch.tensor([[cos_y, 0, sin_y],
                          [0, 1, 0],
                          [-sin_y, 0, cos_y]], device=data.pos.device)

        Rz = torch.tensor([[cos_z, -sin_z, 0],
                          [sin_z, cos_z, 0],
                          [0, 0, 1]], device=data.pos.device)

        R = torch.mm(torch.mm(Rz, Ry), Rx)
        
        # Apply rotation
        data.pos = torch.mm(data.pos, R.t())
        if hasattr(data, 'normal'):
            data.normal = torch.mm(data.normal, R.t())
            
        return data
    
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def scatter_mean(src, index, dim, dim_size):
    # Step 1: Perform scatter add (sum)
    out_shape = [dim_size] + list(src.shape[1:])
    out_sum = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    dims_to_add = src.dim() - index.dim()
    for _ in range(dims_to_add):
        index = index.unsqueeze(-1)
    index_expanded = index.expand_as(src)
    out_sum.scatter_add_(dim, index_expanded, src)
    
    # Step 2: Count occurrences of each index to calculate the mean
    ones = torch.ones_like(src)
    out_count = torch.zeros(out_shape, dtype=torch.float, device=src.device)
    out_count.scatter_add_(dim, index_expanded, ones)
    out_count[out_count == 0] = 1  # Avoid division by zero
    
    # Calculate mean by dividing sum by count
    out_mean = out_sum / out_count

    return out_mean

def fully_connected_edge_index(batch_idx):
    edge_indices = []
    for batch_num in torch.unique(batch_idx):
        # Find indices of nodes in the current batch
        node_indices = torch.where(batch_idx == batch_num)[0]
        grid = torch.meshgrid(node_indices, node_indices, indexing='ij')
        edge_indices.append(torch.stack([grid[0].reshape(-1), grid[1].reshape(-1)], dim=0))
    edge_index = torch.cat(edge_indices, dim=1)
    return edge_index

def subtract_mean(pos, batch):
    means = scatter_mean(src=pos, index=batch, dim=0, dim_size=batch.max().item()+1)
    return pos - means[batch]


class RandomSOd(torch.nn.Module):
        def __init__(self, d):
            """
            Initializes the RandomRotationGenerator.
            Args:
            - d (int): The dimension of the rotation matrices (2 or 3).
            """
            super(RandomSOd, self).__init__()
            assert d in [2, 3], "d must be 2 or 3."
            self.d = d

        def forward(self, n=None):
            """
            Generates random rotation matrices.
            Args:
            - n (int, optional): The number of rotation matrices to generate. If None, generates a single matrix.
            
            Returns:
            - Tensor: A tensor of shape [n, d, d] containing n rotation matrices, or [d, d] if n is None.
            """
            if self.d == 2:
                return self._generate_2d(n)
            else:
                return self._generate_3d(n)
        
        def _generate_2d(self, n):
            theta = torch.rand(n) * 2 * torch.pi if n else torch.rand(1) * 2 * torch.pi
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
            if n:
                return rotation_matrix.view(n, 2, 2)
            return rotation_matrix.view(2, 2)

        def _generate_3d(self, n):
            q = torch.randn(n, 4) if n else torch.randn(4)
            q = q / torch.norm(q, dim=-1, keepdim=True)
            q0, q1, q2, q3 = q.unbind(-1)
            rotation_matrix = torch.stack([
                1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
                2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1),
                2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)
            ], dim=-1)
            if n:
                return rotation_matrix.view(n, 3, 3)
            return rotation_matrix.view(3, 3)

class RandomSO2AroundAxis(torch.nn.Module):
    def __init__(self, axis=2, degrees=15):
        """
        Initializes a generator for rotations around a specific axis.
        Args:
        - axis (int): The rotation axis (0=X, 1=Y, 2=Z)
        - degrees (float or tuple): Maximum rotation angle in degrees.
                                  If float, uses (-|degrees|, |degrees|).
                                  If tuple, uses (min_degrees, max_degrees).
        """
        super().__init__()
        assert axis in [0, 1, 2], "axis must be 0 (X), 1 (Y), or 2 (Z)"
        self.axis = axis
        
        # Handle degrees argument
        if isinstance(degrees, (float, int)):
            self.degrees = (-abs(float(degrees)), abs(float(degrees)))
        elif isinstance(degrees, (tuple, list)):
            assert len(degrees) == 2, "degrees tuple must have length 2"
            self.degrees = tuple(map(float, degrees))
        else:
            raise ValueError("degrees must be a number or a tuple")

    def forward(self, n=None):
        """
        Generates random rotation matrices around the specified axis.
        Args:
        - n (int, optional): The number of rotation matrices to generate. 
                            If None, generates a single matrix.
        
        Returns:
        - Tensor: A tensor of shape [n, 3, 3] containing n rotation matrices, 
                 or [3, 3] if n is None.
        """
        # Generate random angles in degrees
        min_deg, max_deg = self.degrees
        angles = torch.rand(n if n else 1) * (max_deg - min_deg) + min_deg
        # Convert to radians
        theta = angles * torch.pi / 180.0
        cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
        
        # Create rotation matrices based on axis
        if self.axis == 0:  # X-axis
            rotation_matrix = torch.stack([
                torch.ones_like(cos_theta), torch.zeros_like(cos_theta), torch.zeros_like(cos_theta),
                torch.zeros_like(cos_theta), cos_theta, sin_theta,
                torch.zeros_like(cos_theta), -sin_theta, cos_theta
            ], dim=-1)
        elif self.axis == 1:  # Y-axis
            rotation_matrix = torch.stack([
                cos_theta, torch.zeros_like(cos_theta), -sin_theta,
                torch.zeros_like(cos_theta), torch.ones_like(cos_theta), torch.zeros_like(cos_theta),
                sin_theta, torch.zeros_like(cos_theta), cos_theta
            ], dim=-1)
        else:  # Z-axis
            rotation_matrix = torch.stack([
                cos_theta, sin_theta, torch.zeros_like(cos_theta),
                -sin_theta, cos_theta, torch.zeros_like(cos_theta),
                torch.zeros_like(cos_theta), torch.zeros_like(cos_theta), torch.ones_like(cos_theta)
            ], dim=-1)
        
        if n:
            return rotation_matrix.view(n, 3, 3)
        return rotation_matrix.view(3, 3)

class TimerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.total_training_start_time = 0.0
        self.epoch_start_time = 0.0
        self.test_inference_time = 0.0

    # Called when training begins
    def on_train_start(self, trainer, pl_module):
        self.total_training_start_time = time.time()

    # Called when training ends
    def on_train_end(self, trainer, pl_module):
        total_training_time = (time.time() - self.total_training_start_time)/60
        # Log total training time at the end of training
        trainer.logger.experiment.log({"Total Training Time (min)" : total_training_time})

    # Called at the start of the test epoch
    def on_test_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()

    # Called at the end of the test epoch
    def on_test_epoch_end(self, trainer, pl_module):
        # Calculate the inference time for the entire test epoch
        self.test_inference_time = (time.time() - self.epoch_start_time)/60
        # Log the inference time for the test epoch
        trainer.logger.experiment.log({"Test Inference Time (min)": self.test_inference_time})

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing scheduler with warmup."""
    
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= (epoch + 1e-6) / (self.warmup + 1e-6)
        return lr_factor
