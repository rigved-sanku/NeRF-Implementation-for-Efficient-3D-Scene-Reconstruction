import argparse
import glob
from utils import *
from tqdm import tqdm
import random
from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
import os
# Import all the good stuff
from typing import Optional
import numpy as np
import sys
from NeRFModel import *




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('Phase2/tiny_nerf_experiment')

print(device)
np.random.seed(0)
torch.manual_seed(0)

# Don't generate pyc codes
sys.dont_write_bytecode = True


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
  r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

  Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

  Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
  """
  # TESTED
  # Only works for the last dimension (dim=-1)
  dim = -1
  # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
  cumprod = torch.cumprod(tensor, dim)
  # "Roll" the elements along dimension 'dim' by 1 element.
  cumprod = torch.roll(cumprod, 1, dim)
  # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
  cumprod[..., 0] = 1.

  return cumprod

def get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor):
  r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

  Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

  Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
  """

  ii, jj = meshgrid_xy(
      torch.arange(width).to(tform_cam2world),
      torch.arange(height).to(tform_cam2world)
  )
  directions = torch.stack([(ii - width * .5) / focal_length,
                            -(jj - height * .5) / focal_length,
                            -torch.ones_like(ii)
                           ], dim=-1)
  ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
  ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
  return ray_origins, ray_directions

def compute_query_points_from_rays(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near_thresh: float,
    far_thresh: float,
    num_samples: int,
    randomize: Optional[bool] = True
) -> (torch.Tensor, torch.Tensor):
  r"""Compute query 3D points given the "bundle" of rays. The near_thresh and far_thresh
  variables indicate the bounds within which 3D points are to be sampled.

  Args:
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    ray_directions (torch.Tensor): Direction of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    near_thresh (float): The 'near' extent of the bounding volume (i.e., the nearest depth
      coordinate that is of interest/relevance).
    far_thresh (float): The 'far' extent of the bounding volume (i.e., the farthest depth
      coordinate that is of interest/relevance).
    num_samples (int): Number of samples to be drawn along each ray. Samples are drawn
      randomly, whilst trying to ensure "some form of" uniform spacing among them.
    randomize (optional, bool): Whether or not to randomize the sampling of query points.
      By default, this is set to `True`. If disabled (by setting to `False`), we sample
      uniformly spaced points along each ray in the "bundle".

  Returns:
    query_points (torch.Tensor): Query points along each ray
      (shape: :math:`(width, height, num_samples, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).
  """

  # shape: (num_samples)
  depth_values = torch.linspace(near_thresh, far_thresh, num_samples).to(ray_origins)
  if randomize is True:
    # ray_origins: (width, height, 3)
    # noise_shape = (width, height, num_samples)
    noise_shape = list(ray_origins.shape[:-1]) + [num_samples]
    # depth_values: (num_samples)
    depth_values = depth_values \
        + torch.rand(noise_shape).to(ray_origins) * (far_thresh
            - near_thresh) / num_samples
  # (width, height, num_samples, 3) = (width, height, 1, 3) + (width, height, 1, 3) * (num_samples, 1)
  # query_points:  (width, height, num_samples, 3)
  query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., :, None]
  # TODO: Double-check that `depth_values` returned is of shape `(num_samples)`.
  return query_points, depth_values


def render_volume_density(
    rbg_field: torch.Tensor,
    density_field: torch.Tensor,
    ray_origins: torch.Tensor,
    depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
  r"""Differentiably renders a radiance field, given the origin of each ray in the
  "bundle", and the sampled depth values along them.

  Args:
    radiance_field (torch.Tensor): A "field" where, at each query location (X, Y, Z),
      we have an emitted (RGB) color and a volume density (denoted :math:`\sigma` in
      the paper) (shape: :math:`(width, height, num_samples, 4)`).
    ray_origins (torch.Tensor): Origin of each ray in the "bundle" as returned by the
      `get_ray_bundle()` method (shape: :math:`(width, height, 3)`).
    depth_values (torch.Tensor): Sampled depth values along each ray
      (shape: :math:`(num_samples)`).

  Returns:
    rgb_map (torch.Tensor): Rendered RGB image (shape: :math:`(width, height, 3)`).
    depth_map (torch.Tensor): Rendered depth image (shape: :math:`(width, height)`).
    acc_map (torch.Tensor): # TODO: Double-check (I think this is the accumulated
      transmittance map).
  """

  one_e_10 = torch.tensor([1e10], dtype=ray_origins.dtype, device=ray_origins.device)
  dists = torch.cat((depth_values[..., 1:] - depth_values[..., :-1],
                  one_e_10.expand(depth_values[..., :1].shape)), dim=-1)

  alpha = 1. - torch.exp(-density_field * dists)
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

  rgb_map = (weights[..., None] * rbg_field).sum(dim=-2)
  depth_map = (weights * depth_values).sum(dim=-1)
  acc_map = weights.sum(-1)

  return rgb_map, depth_map, acc_map

def get_minibatches(inputs: torch.Tensor, chunksize: Optional[int] = 1024 * 8):
  r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
  Each element of the list (except possibly the last) has dimension `0` of length
  `chunksize`.
  """
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


# One iteration of TinyNeRF (forward pass).
def run_one_iter_of_tinynerf(height, width, focal_length, tform_cam2world,
                             near_thresh, far_thresh, depth_samples_per_ray,
                            get_minibatches_function):

  ray_origins = []
  ray_directions = []
  # Get the "bundle" of rays through all image pixels.
  for i in range(tform_cam2world.shape[0]):
    #print(i)
    r_o, r_d = get_ray_bundle(height, width, focal_length,
                                                 tform_cam2world[i])
    ray_origins.append(r_o)
    ray_directions.append(r_d)

  ray_origins = torch.stack(ray_origins, dim=0)
  ray_directions = torch.stack(ray_directions, dim=0)

  # Sample query points along each ray
  query_points, depth_values = compute_query_points_from_rays(
      ray_origins, ray_directions, near_thresh, far_thresh, depth_samples_per_ray
  )
  #(query_points.shape, depth_values.shape, ray_origins.shape, ray_directions.shape)
  # "Flatten" the query points.
  flattened_query_points = query_points.reshape((-1, 3)).to(device)

  # "Flatten" the ray direction points and convert to shape as falltened querry points
  exapanded_ray_directions = ray_directions.expand(depth_samples_per_ray, ray_directions.shape[0], ray_directions.shape[1], ray_directions.shape[2], 3).permute(1,2,3,0,4)
  flattened_ray_direction = exapanded_ray_directions.reshape((-1,3)).to(device)


  # Split the encoded points into "chunks", run the model on all chunks, and
  # concatenate the results (to avoid out-of-memory issues).
  batches_pos = get_minibatches_function(flattened_query_points, chunksize=chunksize)
  batches_dir = get_minibatches_function(flattened_ray_direction, chunksize=chunksize)

  rgb = []
  density = []
  for batch_pos, batch_dir in zip(batches_pos, batches_dir):
    # Forward pass
    color, sigma = model(batch_pos, batch_dir)
    rgb.append(color)
    density.append(sigma)

  rgb_flattened = torch.cat(rgb, dim=0)
  density_flattened  = torch.cat(density, dim=0)

  # "Unflattening"
  unflattened_shape = list(query_points.shape[:-1]) + [3]
  rbg_field = torch.reshape(rgb_flattened, unflattened_shape)

  unflattened_shape = list(query_points.shape[:-1])
  density_field = torch.reshape(density_flattened, unflattened_shape)

  # Perform differentiable volume rendering to re-synthesize the RGB image.
  rgb_predicted, _, _ = render_volume_density(rbg_field, density_field, ray_origins, depth_values)

  return rgb_predicted

def plot_figures(Epochs, log_loss):
    plt.figure(figsize=(10, 4))
    plt.plot(Epochs, log_loss)
    plt.title("loss")
    # plt.show()0
    plt.savefig("Loss.png")
    

lego_dataset = NeRFDatasetLoader('Phase2/data/ship', 'train')
focal_length, tform_cam2world, images = lego_dataset.get_full_data(device)
images = images / 255.0

height, width = images.shape[1:3]

# Hold one image out (for test).
testimg, testpose = images[99], tform_cam2world[99]
testpose = testpose.unsqueeze(0)
# Near and far clipping thresholds for depth values.
near_thresh = 2.
far_thresh = 6.

# Map images to device
images = images[:98, ..., :3]

# Optimizer parameters
lr = 5e-4
num_iters = 4096 # Number of iterations to train for

# Number of depth samples along each ray.
depth_samples_per_ray = 32
chunksize = 4096 # Number of points to process in each minibatch

# Misc parameters
display_every = 100  # Number of iters after which stats are displayed

"""
Model
"""
model = NeRFmodel(hidden_dim=128, use_encoding= True)
model.to(device)

"""
Optimizer
"""
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[187, 375, 750], gamma=0.5)

"""
Train-Eval-Repeat!
"""

# Lists to log metrics etc.
psnrs = []
iternums = []
scaler = torch.cuda.amp.GradScaler()

start_iter = 0
save_every = 100  # Save a checkpoint every 100 iterations
checkpoint_dir = 'Phase2/checkpoints'  # Directory to save checkpoints

load_checkpoint_dir = 'Phase2/checkpoints'

if load_checkpoint_dir is not None and False:
  checkpoint = torch.load('Phase2/checkpoints/checkpoint_1000.pt')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_iter = checkpoint['iteration']
  scheduler.load_state_dict(checkpoint['scheduler'])
  scaler.load_state_dict(checkpoint['scaler'])  # Load scaler state
  
  print(f"Loaded checkpoint from {load_checkpoint_dir}, starting from iteration {start_iter}")
  
else:
  start_iter = 0
  print("No checkpoint found, starting from scratch")

Loss =[]
Epochs = []

for i in range(start_iter, num_iters):
  model.train()
  with torch.cuda.amp.autocast():
    # Randomly pick an image as the target.
    target_img_idx = np.random.choice(images.shape[0], 4)
    target_img = images[target_img_idx]
    target_tform_cam2world = tform_cam2world[target_img_idx].to(device)
    #print(target_tform_cam2world.shape)

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
                                            target_tform_cam2world, near_thresh,
                                            far_thresh, depth_samples_per_ray,
                                            get_minibatches)
    
    rgb_predicted = rgb_predicted.squeeze(0)
    # Compute mean-squared error between the predicted and target images. Backprop!
    loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
    
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  optimizer.zero_grad(set_to_none=True)
  scheduler.step()
  #loss.backward()
  #optimizer.step()
  #optimizer.zero_grad()

  print(f"Iteration: {i}, Loss: {loss.item()}")
  
  if i% save_every == 0:
    Loss.append(loss.item())
    Epochs.append(i+1)
  
    
    print("Saving Checkpoint")
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'iteration': i,
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict()
            }, f"{checkpoint_dir}/checkpoint_{i}.pt")
    plot_figures(Epochs, Loss)
    
    
  # Display images/plots/stats
  # if i % display_every == 0:
  #   # Render the held-out view
  #   model.eval()
  #   rgb_predicted = run_one_iter_of_tinynerf(height, width, focal_length,
  #                                           testpose, near_thresh,
  #                                           far_thresh, depth_samples_per_ray,
  #                                           get_minibatches)

  #   #loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
  #   #print("Loss:", loss.item())
  #   #psnr = -10. * torch.log10(loss)

  #   #psnrs.append(psnr.item())
  #   #iternums.append(i)
  #   plt.figure(figsize=(10, 4))
  #   plt.subplot(121)
  #   plt.imshow(rgb_predicted.detach().cpu().numpy())
  #   plt.title(f"Iteration {i}")
  #   plt.subplot(122)
  #   #plt.plot(iternums, psnrs)
  #   #plt.title("PSNR")
  #   plt.show()

  
plot_figures(Epochs, Loss)
print('Done!')


