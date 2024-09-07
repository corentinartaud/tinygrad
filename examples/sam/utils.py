from typing import Tuple, List, Generator, Any
import math
from itertools import product
from copy import deepcopy
from PIL import Image
import numpy as np

def generate_crop_boxes(img_size:Tuple[int, ...], n_layers:int, overlap_ratio:float) -> Tuple[List[List[int]], List[int]]:
  """Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes of the ith layer."""
  img_h, img_w = img_size
  short_side = min(img_h, img_w)
  def crop_len(orig_len, n_crops, overlap): return int(math.ceil((overlap * (n_crops - 1) + orig_len) / n_crops))
  crop_boxes, layer_idxs = [[0, 0, img_w, img_h]], [0] # Original image
  for i_layer in range(n_layers):
    n_crops_per_side = 2 ** (i_layer + 1)
    overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))
    _crop_len = lambda img_side: crop_len(img_side, n_crops_per_side, overlap)
    crop_w, crop_h = _crop_len(img_w), _crop_len(img_h)
    crop_box_x, crop_box_y = zip(*[[int((crop_w - overlap) * i), int((crop_h - overlap * i))] for i in range(n_crops_per_side)])
    for x0, y0 in product(crop_box_x, crop_box_y): # Crops in XYWH format
      crop_boxes.append([x0, y0, min(x0 + crop_w, img_w), min(y0 + crop_h, img_h)])
      layer_idxs.append(i_layer + 1)
  return crop_boxes, layer_idxs

def build_point_grid(n_per_side:int) -> np.ndarray:
  """Generates a 2D grid of points evenly spaces in [0,1]x[0,1]."""
  offset = 1 / (2 * n_per_side)
  points_one_side = np.linspace(offset, 1 - offset, n_per_side)
  points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
  points_y = np.tile(points_one_side[:, None], (1, n_per_side))
  points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
  return points

def build_all_layer_point_grids(n_per_side:int, n_layers:int, scale_per_layer:int) -> List[np.ndarray]:
  """Generates point grids for all crop layers."""
  return [build_point_grid(int(n_per_side / (scale_per_layer**i))) for i in range(n_layers+1)]

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
  assert len(args) > 0 and all(len(a) == len(args[0]) for a in args), \
    "Batched iteration must have inputs of all the same size."
  n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
  for b in range(n_batches): yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

class ResizeLongestSide:
  def __init__(self, target_length:int) -> None:
    self.target_length = target_length
  
  def apply_image(self, image:np.ndarray) -> np.ndarray:
    """Expects a numpy array with shape HxWxC in uint8 format."""
    target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
    # TODO: Check if this the right resize or shoule use target_size[::-1]
    return np.array(Image.fromarray(image).resize(target_size, Image.BILINEAR))
  
  def apply_coords(self, coords:np.ndarray, orig_size:Tuple[int,...]) -> np.ndarray:
    """Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format."""
    oldh, oldw = orig_size
    newh, neww = self.get_preprocess_shape(oldh, oldw, self.target_length)
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (neww / oldw)
    coords[..., 1] = coords[..., 1] * (newh / oldh)
    return coords
  
  @staticmethod
  def get_preprocess_shape(oldh:int, oldw:int, long_side_length:int) -> Tuple[int, int]:
    """Compute the output size given input size and target long side length."""
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    return (int(neww + 0.5), int(newh + 0.5))