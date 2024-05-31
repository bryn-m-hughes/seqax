# To run:
# 
# ```
# python write_synthetic_dataset.py --config-name=synthetic_dataset +output=synthetic_dataset.zarr
# ```
#
# Random dots task

from functools import partial
import hydra
from jaxtyping import Float, Int, jaxtyped, UInt32, UInt8
import numpy as np
from typeguard import typechecked as typechecker
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
import flat_vl_tokens
import math
from constants import IMG_TOKENS

@dataclass
class Config:
  output: str
  seed: int
  seq_len: int
  examples: int
  border_length: int
  flat_tokens_config: flat_vl_tokens.Config
  _target_: str = __name__ + '.Config'

def reading_squares(targets: list[UInt32[np.ndarray, '...']], colors: tuple[int, int, int], border_length: int) -> list[UInt8[np.ndarray, '...']]:
    images = []
    for target in targets:
        image = np.ones((border_length, border_length, 3), dtype=np.uint8) * 255
        num_squares = len(target)
        squares_per_side = math.ceil(math.sqrt(num_squares))
        square_length = border_length // squares_per_side
        for i, color_index in enumerate(target):
            color = colors[color_index % len(colors)]
            x_start = (i % squares_per_side) * square_length
            y_start = (i // squares_per_side) * square_length
            
            x_end = x_start + square_length
            y_end = y_start + square_length
            
            image[x_start:x_end, y_start:y_end] = color

        images.append(image)
    return images
    

def synthetic_task(config: Config, gen: np.random.Generator) -> tuple[list[UInt32[np.ndarray, '...']], list[UInt8[np.ndarray, '...']]]:
  colors = [
        (0, 0, 0),       # black
        (255, 0, 0),     # red
        (0, 255, 0),     # green
        (255, 255, 0),   # yellow
        (0, 0, 255),     # blue
        (255, 0, 255),   # magenta
        (0, 255, 255)    # cyan
    ]
  num_colors = len(colors)

  targets_len = config.seq_len-len(IMG_TOKENS)
  targets = [gen.integers(1, num_colors, (targets_len,), dtype=np.uint32) for i in range(config.examples)]
  image_data = reading_squares(targets, colors, config.border_length)
  targets = [np.concatenate([IMG_TOKENS, seq]) for seq in targets]

  return targets, image_data


# Registering the Config class with the name 'config'.
ConfigStore.instance().store(name="config_schema", node=Config)


@hydra.main(config_path="configs", version_base=None)
def main(config):
  config = hydra.utils.instantiate(config)
  gen = np.random.Generator(np.random.PCG64(config.seed))
  for split, mode in [(flat_vl_tokens.Split.VALIDATION, "w-"), (flat_vl_tokens.Split.TRAIN, "r+")]:
    dst = flat_vl_tokens.Writer(config.output, split, mode, config.flat_tokens_config)
    examples, images = synthetic_task(config, gen)
    
    from_ragged = flat_vl_tokens.Chunk.from_ragged(examples, images)
    dst.write(from_ragged)

if __name__ == "__main__":
  main()