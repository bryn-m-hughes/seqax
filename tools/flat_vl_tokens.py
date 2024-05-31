from dataclasses import dataclass
import enum
import zarr
import numpy as np
import concurrent
from numcodecs import Blosc, Delta
from constants import IMG_INDEX_TOKEN

class Split(enum.Enum):
  TRAIN = "train"
  VALIDATION = "validation"

@dataclass
class Config:
  tokens_chunk_size: int
  seq_starts_chunk_size: int
  images_chunk_size: int
  _target_: str = __name__ + ".Config"


@dataclass
class Chunk:
  """An in-memory encoding of flat vl data. Use this as a buffer for writing to a FlatVLWriter."""
  encoded_tokens: np.ndarray  # uint32[num_tokens]
  seq_starts: np.ndarray  # uint64[num_seqs]
  images: np.ndarray  # uint8[num_seqs*border_length*border_length*3]
  max_token_id: int

  @staticmethod
  def from_ragged(sequences: list[np.ndarray], images: list[np.ndarray]):
    """Converts a list of sequences to a FlatTokensChunk."""
    tokens = np.concatenate(sequences)
    max_token_id = tokens.max()
    image_indices = np.where(tokens == IMG_INDEX_TOKEN)[0]
    tokens[image_indices] = np.arange(len(image_indices))
    seq_starts = np.zeros(len(sequences) + 1, np.uint64)
    np.cumsum([len(seq) for seq in sequences], out=seq_starts[1:])
    images = np.concatenate([img.flatten() for img in images])
    # Some number of the seq_starts will equal len(tokens). Typically it's just one of them,
    # but if `sequences` ends with some empty sequences there may be more. Avoid out-of-bounds
    # accesses to `tokens`.
    in_bounds_seq_starts = np.where(seq_starts != len(tokens), seq_starts, 0)
    tokens <<= 1
    tokens[in_bounds_seq_starts] |= 1
    return Chunk(tokens, seq_starts, images, max_token_id)


class Writer:
  def __init__(self, filespec: str, split: Split, mode: str, config: Config):
    try:
        dst_root = zarr.open_group(filespec, mode=mode, cache_attrs=True)
    except zarr.errors.ContainsGroupError:
        raise ValueError(f"Output {filespec} already exists.")
    self.group = dst_root.require_group(split.value)

    # Use BITSHUFFLE for encoded_tokens, since the token IDs will typically only be ~14-17 bits wide.
    compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)

    if "max_token_id" not in self.group.attrs:
      self.group.attrs["max_token_id"] = 0

    if "encoded_tokens" in self.group:
      self.encoded_tokens = self.group["encoded_tokens"]
    else:
      self.encoded_tokens = self.group.empty("encoded_tokens", shape=(0,), chunks=(config.tokens_chunk_size,), dtype=np.uint32, compressor=compressor)

    if "seq_starts" in self.group:
      self.seq_starts = self.group["seq_starts"]
    else:
      # Use delta encoding for seq_starts, since they're known to be sorted.
      filters = [Delta(dtype='i8')]
      self.seq_starts = self.group.zeros("seq_starts", shape=(1,), chunks=(config.seq_starts_chunk_size,), dtype=np.uint64, compressor=compressor, filters=filters)

    if "images" in self.group:
      self.images = self.group["images"]
    else:
      self.images = self.group.empty("images", shape=(0,), chunks=(config.images_chunk_size), dtype=np.uint8, compressor=compressor)

  def write(self, chunk: Chunk):
    """Synchronously writes a chunk of flat tokens to the underlying storage.

    Results are committed when this function returns, no need for a separate close() or 
    flush() call. This function should not be called concurrently for the same destination.
    
    You typically want to call this in a separate thread, to overlap computation with I/O.
    """
    num_tokens = self.encoded_tokens.shape[0]
    if chunk.max_token_id > self.group.attrs["max_token_id"]:
      self.group.attrs["max_token_id"] = chunk.max_token_id
    # In parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      executor.submit(lambda: self.encoded_tokens.append(chunk.encoded_tokens))
      executor.submit(lambda: self.seq_starts.append(num_tokens + chunk.seq_starts[1:]))
      executor.submit(lambda: self.images.append(chunk.images))