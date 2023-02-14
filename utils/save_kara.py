import os
import json
import kara_storage
from tqdm import tqdm

out_path = "/data1/private/xiaochaojun/ChineseIR/ChineseQAData/kara/dureader"
if os.path.exists(out_path):
    os.system("rm -rf %s" % out_path)
os.makedirs(out_path, exist_ok=True)

storage = kara_storage.KaraStorage("file://%s" % out_path)
dataset = storage.open_dataset("dureader", "train", "w", version="lastest")
validdataset = storage.open_dataset("dureader", "valid", "w", version="lastest")

fin = open("/data1/private/xiaochaojun/ChineseIR/ChineseQAData/dureader-IR/train.jsonl", "r")
for line in tqdm(fin):
    dataset.write(json.loads(line))
fin.close()

fin = open("/data1/private/xiaochaojun/ChineseIR/ChineseQAData/dureader-IR/dev.jsonl", "r")
for line in tqdm(fin):
    validdataset.write(json.loads(line))
fin.close()

dataset.close()
validdataset.close()
