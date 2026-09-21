from pathlib import Path

from datasets import load_dataset, load_from_disk, Dataset

_default_path = './data'

def download_usps(path: str = _default_path): 
    ds = load_dataset('flwrlabs/usps')
    ds.save_to_disk(str(path))

def get_usps(path: str = _default_path) -> Dataset: 
    return load_from_disk(str(path))
