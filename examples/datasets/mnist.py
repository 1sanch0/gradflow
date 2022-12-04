from gradflow import Tensor
from pathlib import Path
from tqdm import tqdm
import numpy as np
import requests
import gzip

__all__ = ["MNISTDataset"]

TRAIN_IMGS   = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
TRAIN_LABELS = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
TEST_IMGS    = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
TEST_LABELS  = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
URLS = [TRAIN_IMGS, TEST_IMGS, TRAIN_LABELS, TEST_LABELS]

def download(path: str, chunk_size: int = 8192) -> None:
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  
  for url in URLS:
    filename = url.split("/")[-1]

    if (path / filename).exists():
      print(f"{filename} already exists in {path}. Skipping...")
      continue

    response = requests.get(url, stream=True)
    with open(path / filename, "wb") as f:
      for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
        f.write(chunk)

def load(path: str, url: str) -> np.ndarray:
  path = Path(path)

  filename = url.split("/")[-1]
  offset = 16 if url in URLS[:2] else 8 # ¯\_(ツ)_/¯ idk

  with gzip.open(path / filename, "rb") as f:
    return np.frombuffer(f.read(), offset=offset, dtype=np.uint8).astype(np.uint8)


class MNISTDataset:
  def __init__(self, path: str, batch_size: int = 32, train: bool = True, flatten: bool = True):
    download(path)

    if train:
      imgs   = load(path, TRAIN_IMGS)
      labels = load(path, TRAIN_LABELS)
    else:
      imgs   = load(path, TEST_IMGS)
      labels = load(path, TEST_LABELS)
    
    if flatten:
      imgs = imgs.reshape(-1, batch_size, 28*28) 
    else:
      imgs = imgs.reshape(-1, batch_size, 28,28) 
    
    # Preprocessing
    self.imgs = imgs.astype(np.float32) / 255.0
    # self.imgs -= np.mean(self.imgs, axis=0)
    # self.imgs /= np.std(self.imgs, axis=0)
    # print(np.mean(self.imgs, axis=0).shape)
    # print(np.mean(self.imgs, axis=1).shape)
    # print(np.mean(self.imgs, axis=2).shape)
    # print(np.mean(self.imgs, axis=3).shape)
    


    self.labels = np.eye(10)[labels].reshape(-1, batch_size, 10)
  
  def __getitem__(self, i: int):
    return Tensor(self.imgs[i]), Tensor(self.labels[i])
  
  def __len__(self) -> int:
    return len(self.labels)  
  


