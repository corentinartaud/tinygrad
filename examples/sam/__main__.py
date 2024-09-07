import argparse
import io, cv2, copy
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from tinygrad import Tensor, dtypes
from tinygrad.helpers import fetch
import tinygrad.nn as nn
from examples.sam import utils
from rich import print

class PatchEmbed:
  def __init__(
    self, 
    kernel_size: Tuple[int, int] = (16, 16), 
    stride: Tuple[int, int] = (16, 16),
    padding: Tuple[int, int] = (0, 0),
    in_chans: int = 3,
    embed_dim: int = 768,
  ) -> None:
    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, stride, padding)
  
  def __call__(self, x: Tensor) -> Tensor:
    return self.proj(x).permute(0, 2, 3, 1) # B C H W -> B H W C
  
class Block:
  def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, norm_layer=nn.LayerNorm, act_layer=Tensor.gelu, 
    use_rel_pos=False, rel_pos_zero_init=True, window_size=0, input_size=(0, 0)) -> None:
    pass

class ImageEncoderViT:
  def __init__(
    self,
    img_size: int = 1024,
    patch_size: int = 16,
    in_chans: int = 3,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    out_chans: int = 256,
    qkv_bias: bool = True,
    norm_layer = nn.LayerNorm,
    act_layer = Tensor.gelu,
    use_abs_pos: bool = True,
    use_rel_pos: bool = False,
    rel_pos_zero_init: bool = True,
    window_size: int = 0,
    global_attn_indexes: Tuple[int, ...] = (),
  ) -> None:
    self.img_size = img_size
    self.patch_embed = PatchEmbed((patch_size, patch_size), (patch_size, patch_size), in_chans=in_chans, embed_dim=embed_dim)
    self.pos_embed: Optional[Tensor] = None
    # Initialize absolute positional embedding with pretrain image size
    if use_abs_pos: self.pos_embed = Tensor.zeros(1, img_size//patch_size, img_size//patch_size, embed_dim)
    self.blocks = [
      Block(embed_dim, num_heads, mlp_ratio, qkv_bias, norm_layer, act_layer, use_rel_pos, rel_pos_zero_init, 
        window_size=(window_size if i not in global_attn_indexes else 0), 
        input_size=(img_size//patch_size, img_size//patch_size))
      for i in range(depth)
    ]
    # NOTE: Compare against source implementation of LayerNorm2d in segment-anything
    self.neck = [
      nn.Conv2d(embed_dim, out_chans, 1, bias=False), nn.LayerNorm2d(out_chans), 
      nn.Conv2d(out_chans, out_chans, 3, padding=1, bias=False), nn.LayerNorm2d(out_chans)]
  
  def __call__(self, x:Tensor) -> Tensor:
    x = self.patch_embed(x)
    if self.pos_embed is not None: x = x + self.pos_embed
    x = x.permute(0, 3, 1, 2).sequential(self.neck)
    return x

# ****** Prompt Encoder ******

# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L171
class PositionEmbeddingRandom:
  """Positional encoding using random spatial frequencies."""
  def __init__(self, num_pos_feats:int=64, scale:Optional[float]=None) -> None:
    if scale is None or scale <= 0.0: scale = 1.0
    self.positional_encoding_gaussian_matrix = scale * Tensor.randn((2, num_pos_feats))

  def _pe_encoding(self, coords:Tensor) -> Tensor:
    """Positionally encode points that are normalized to [0, 1]."""
    # Assuming coords are in [0, 1]^2 square and have d_1, x ... x d_n x 2 shape
    coords = 2 * coords - 1
    coords = coords @ self.positional_encoding_gaussian_matrix
    coords = 2 * np.pi * coords
    # Outputs d_1 x ... x d_n x C tensor
    return (coords.sin()).cat(coords.cos(), dim=-1)

  def forward(self, coords_input:Tensor, img_size:Tuple[int, int]) -> Tensor:
    """Positionally encode points that are not normalized to [0,1]."""
    coords = copy.copy(coords_input)
    coords[:, :, 0] = coords[:, :, 0] / img_size[1]
    coords[:, :, 1] = coords[:, :, 1] / img_size[0]
    return self._pe_encoding(coords.cast(dtypes.float))  # B x N x C
  
  def __call__(self, size:Tuple[int, int]) -> Tensor:
    """Generate positional encoding for a grid of the specified size."""
    grid = Tensor.ones(size, device=self.positional_encoding_gaussian_matrix.device, dtype=dtypes.float)
    y_embed = (grid.cumsum(axis=0) - 0.5) / size[0]
    x_embed = (grid.cumsum(axis=1) - 0.5) / size[1]
    pe = self._pe_encoding(x_embed.stack(y_embed, axis=-1))
    return pe.permute(2, 0, 1) # C x H x W

# https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L16
class PromptEncoder:
  def __init__(self, embed_dim:int, image_embedding_size:Tuple[int, int], input_image_size:Tuple[int, int], mask_in_chans:int):
    """Encodes prompts for input to SAM's mask decoder"""
    self.embed_dim, self.input_image_size, self.image_embedding_size = embed_dim, input_image_size, image_embedding_size
    self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
    self.num_point_embeddings: int = 4 # pos/neg points + 2 box corners
    self.point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
    self.not_a_point_embed = nn.Embedding(1, embed_dim)
    self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
    self.mask_downscaling = [
      nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
      nn.LayerNorm2d(mask_in_chans // 4), Tensor.gelu,
      nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
      nn.LayerNorm2d(mask_in_chans), Tensor.gelu,
      nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1)
    ]
    self.no_mask_embed = nn.Embedding(1, embed_dim)
  
  def __call__(self, points:Optional[Tuple[Tensor, Tensor]], boxes:Optional[Tensor], masks:Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    # Gets the batch size of the output given the batch size of the input prompts
    bs = points[0].shape[0] if points is not None \
      else boxes.shape[0] if boxes is not None else masks.shape[0] if masks is not None else 1
    sparse_embeddings = Tensor.empty((bs, 0, self.embed_dim), device=self.point_embeddings[0].weight.device)
    if points is not None:
      coords, labels = points
      coords = coords + 0.5 # Shift to center pixel
      coords = coords.cat(Tensor.zeros((coords.shape[0], 1, 2), device=coords.device), dim=1)
      labels = labels.cat(-Tensor.ones((labels.shape[0], 1), dtype=labels.dtype, device=labels.device), dim=1)
      point_embedding = self.pe_layer.forward(coords, self.input_image_size)
      # NOTE: NotImplementedError: Advanced indexing setitem is not currently supported
      # TODO: Bounty for fix and test - rebranch and refactor
      point_embedding[labels == -1] = 0.0
      point_embedding[labels == -1] += self.not_a_point_embed.weight
      point_embedding[labels == 0] += self.point_embeddings[0].weight
      point_embedding[labels == 1] += self.point_embeddings[1].weight
      sparse_embeddings = sparse_embeddings.cat(point_embedding, dim=1)
    if boxes is not None:
      boxes = boxes + 0.5  # Shift to center of pixel
      coords = boxes.reshape(-1, 2, 2)
      corner_embedding = self.pe_layer.forward(coords, self.input_image_size)
      corner_embedding[:, 0, :] += self.point_embeddings[2].weight
      corner_embedding[:, 1, :] += self.point_embeddings[3].weight
      sparse_embeddings = sparse_embeddings.cat(corner_embedding, dim=1)

    if masks is not None: dense_embeddings = self.mask_downscaling(masks.unsqueeze(1)) # Embeds mask inputs
    else: dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(bs, -1, *self.image_embedding_size)
    return sparse_embeddings, dense_embeddings

class Sam:
  image_format = "RGB"
  def __init__(self, image_encoder, prompt_encoder, mask_decoder, pixel_mean:List[float]=[123.675, 116.28, 103.53], pixel_std:List[float]=[58.395, 57.12, 57.375]) -> None:
    self.image_encoder = image_encoder
    self.prompt_encoder = prompt_encoder
    # self.mask_decoder = mask_decoder
    self.pixel_mean, self.pixel_std = pixel_mean, pixel_std
  
  def __call__(self, x:Tensor) -> Tensor: pass

  def preprocess(self, x:Tensor) -> Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    # NOTE: Is there anyway to make these class varaibles without having the tensors part of the state-dict?
    x = (x - Tensor(self.pixel_mean).view(-1, 1, 1)) / Tensor(self.pixel_std).view(-1, 1, 1)
    # Pad to square
    h, w = x.shape[-2:]
    padh, padw = (self.image_encoder.img_size - h), (self.image_encoder.img_size - w)
    return x.pad(((0, 0), (0, 0), (0, padh), (0, padw)))


class SamPredictor:
  def __init__(self, sam_model:Sam) -> None:
    self.model = sam_model
    self.transform = utils.ResizeLongestSide(sam_model.image_encoder.img_size)
    self.reset_image()

  def set_image(self, image:np.ndarray, image_format:str="RGB") -> None:
    assert image_format in ["RGB", "BGR"], f"image_format must be in ['RGB', 'BGR'], got {image_format}."
    if image_format != self.model.image_format: image = image[..., ::-1]
    # Transform the image to the form expected by the model
    input_image = self.transform.apply_image(image)
    transformed_image = Tensor(input_image).permute(2, 0, 1).contiguous()[None, ...]
    assert (
      len(transformed_image.shape) == 4
      and transformed_image.shape[1] == 3
      and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
    ),f"set_image input must be BCHW with long side {self.model.image_encoder.img_size}"
    self.reset_image()
    self.original_size = image.shape[:2]
    self.input_size = tuple(transformed_image.shape[-2:])
    input_image = self.model.preprocess(transformed_image)
    self.features = self.model.image_encoder(input_image)
    self.is_image_set = True
  
  def predict(self, point_coords, point_labels, boxes=None, mask_input=None, multimask_output:bool=True, return_logits:bool=False):
    if not self.is_image_set: raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
    points = (point_coords, point_labels) if point_coords is not None else None
    sparse_embedding, dense_embedding = self.model.prompt_encoder(points, boxes, mask_input)
  
  def reset_image(self) -> None:
    """Reset the currently set image."""
    self.is_image_set = False
    self.features, self.orig_h, self.orig_w, self.input_h, self.input_w = None, None, None, None, None


class SamAutomaticMaskGenerator:
  def __init__(self, model:Sam, points_per_side:Optional[int]=32, points_per_batch:int=64, 
      crop_n_layers:int=0, crop_overlap_ratio:float=512/1500, crop_n_points_downscale_factor:int=1, 
      point_grids:Optional[List[np.ndarray]]=None) -> None:
    assert (points_per_side is None) != (point_grids is None), \
      "Exactly one of points_per_side or point_grids must be provided."
    if points_per_side is not None: self.point_grids = utils.build_all_layer_point_grids(
      points_per_side, crop_n_layers, crop_n_points_downscale_factor)
    elif point_grids is not None: self.point_grids = point_grids
    else: ValueError("Can't have both points_per_side and point_grids as None.")
    self.predictor = SamPredictor(model)
    self.points_per_batch = points_per_batch
    self.crop_n_layers = crop_n_layers
    self.crop_overlap_ratio = crop_overlap_ratio

  @Tensor.test()
  def generate(self, image:np.ndarray) -> List[Dict[str, Any]]:
    mask_data = self._generate_masks(image)
  
  def _generate_masks(self, image:np.ndarray):
    orig_size = image.shape[:2]
    crop_boxes, layer_idxs = utils.generate_crop_boxes(orig_size, self.crop_n_layers, self.crop_overlap_ratio)
    for crop_box, layer_idxs in zip(crop_boxes, layer_idxs):
      self._process_crop(image, crop_box, layer_idxs, orig_size)
  
  def _process_crop(self, image:np.ndarray, crop_box:List[int], crop_layer_idx:int, orig_size:Tuple[int, ...]):
    # Crop the image and calculate embeddings
    x0, y0, x1, y1 = crop_box
    cropped_img = image[y0:y1, x0:x1, :]
    cropped_img_size = cropped_img.shape[:2]
    self.predictor.set_image(cropped_img)
    # Get points for this crop
    points_scale = np.array(cropped_img_size)[None, ::-1]
    points_per_image = self.point_grids[crop_layer_idx] * points_scale
    for (points,) in utils.batch_iterator(self.points_per_batch, points_per_image):
      batch_data = self._process_batch(points, cropped_img_size, crop_box, orig_size) 
    self.predictor.reset_image()
  
  def _process_batch(self, points:np.ndarray, img_size:Tuple[int, ...], crop_box: List[int], orig_size: Tuple[int, ...]):
    orig_h, orig_w = orig_size
    # Run model on this batch
    transformed_points = self.predictor.transform.apply_coords(points, img_size)
    in_points = Tensor(transformed_points, dtype=dtypes.float32) # TODO: Add DEVICE on this Tensor
    in_labels = Tensor.zeros(in_points.shape[0], dtype=dtypes.int, device=in_points.device)
    masks, iou_preds, _ = self.predictor.predict(
      in_points[:, None, :], in_labels[:, None], multimask_output=True, return_logits=True
    )

MODEL_REGISTRY = {
  "default": {
    "ckpt": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "image_encoder": {
      "embed_dim": 1280,
      "depth": 32,
      "num_heads": 16,
      "global_attn_indexes": [7, 15, 23, 31]
    }
  }
}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Segment Anything Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  weights = fetch("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth")
  weights = nn.state.torch_load(weights)

  prompt_embed_dim = 256
  image_size = 1024
  vit_patch_size = 16
  image_embedding_size = image_size // vit_patch_size
  model = Sam(
    image_encoder=ImageEncoderViT(
      img_size=image_size,
      mlp_ratio=4,
      patch_size=vit_patch_size,
      qkv_bias=True,
      use_rel_pos=True,
      window_size=14,
      out_chans=prompt_embed_dim,
      **MODEL_REGISTRY["default"]["image_encoder"]
    ),
    prompt_encoder=PromptEncoder(
      embed_dim=prompt_embed_dim,
      image_embedding_size=(image_embedding_size, image_embedding_size),
      input_image_size=(image_size, image_size),
      mask_in_chans=16,
    ),
    mask_decoder=None
  )
  nn.state.load_state_dict(model, weights)
  mask_generator = SamAutomaticMaskGenerator(model)

  url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
  img_stream = io.BytesIO(fetch(url).read_bytes())
  image = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  masks = mask_generator.generate(image)
  
  # import matplotlib.pyplot as plt
  # plt.imshow(image)
  # plt.axis('off')
  # plt.show()