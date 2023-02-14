import json
import os
from torch.utils.data import Dataset
from tools import print_rank

class JsonlDataset(Dataset):
    def __init__(self, config, mode, encoding="utf8", *args, **params):
        self.config = config
        self.mode = mode
        
        self.ctxnum = 3
        fin = open(config.get("data", "%s_data_path" % mode), "r")
        self.data = [json.loads(line) for line in fin]
        print_rank("read %s data: %d" % (mode, len(self.data)))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
