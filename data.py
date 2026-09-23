# quoting the paper: The data base used to train and test the network
# consists of 9298 segmented numerals digitized from handwritten zip codes
# ... 7291 examples are used for training the network and 2007 are used for testing
# the generalization performance. 
from datasets import load_dataset, load_from_disk, Dataset

_default_path = './data'

def download_usps(path: str = _default_path): 
    ds = load_dataset('flwrlabs/usps')
    ds.save_to_disk(str(path))

def get_usps(path: str = _default_path) -> Dataset: 
    return load_from_disk(str(path))
