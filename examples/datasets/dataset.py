from typing import Any
from abc import ABC, abstractmethod

import requests
from tqdm import tqdm
from pathlib import Path

def download_from_url(path: str, urls: list[str], chunk_size: int = 8192) -> None:
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  
  for url in urls:
    filename = url.split("/")[-1]

    if (path / filename).exists():
      print(f"{filename} already exists in {path}. Skipping...")
      continue

    response = requests.get(url, stream=True)
    with open(path / filename, "wb") as f:
      for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
        f.write(chunk)

class Dataset(ABC):
  @abstractmethod
  def __len__(self) -> int:
    pass

  @abstractmethod
  def __getitem__(self, index: int) -> Any:
    pass