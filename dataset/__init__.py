from .JsonlDataset import JsonlDataset
from .kara.KaraDataset import make_kara_dataset

dataset_list = {
    "jsonl": JsonlDataset,
    "kara": make_kara_dataset,
}
