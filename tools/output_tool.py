from genericpath import exists
import json
import os
from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return str(data)

def microf1_output_function(data, config, *args, **params):
    if data["TP"] == 0:
        precision, recall, f1 = 0, 0, 0
    else:
        precision = data["TP"] / (data["TP"] + data["FP"])
        recall = data["TP"] / (data["TP"] + data["FN"])
        f1 = 2 * precision * recall / (precision + recall)
    return json.dumps({
        "p": round(precision, 4),
        "r": round(recall, 4),
        "f1": round(f1, 4)
    })

def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)

def output_function1(data, config, *args, **params):
    if data['pre_num'] != 0 and data['actual_num'] != 0:
        pre = data['right'] / data['pre_num']
        recall = data['right'] / data['actual_num']
        if pre + recall == 0:
            f1 = 0
        else:
            f1 = 2 * pre * recall / (pre + recall)
    else:
        pre = 0
        recall = 0
        f1 = 0

    metric = {
            'precision': round(pre, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
        }
    if 'labelset' in data and 'doc_num' in data and data['doc_num'] != 0:
        metric['ave_len'] = data['labelset'] / data['doc_num']
    return json.dumps(metric)

def binary_output_function(data, config, *args, **params):
    if data['total'] == 0:
        metric = {'acc': 0}
    else:
        metric = {'acc': round(data['right'] / data['total'], 4)}
    return json.dumps(metric)

def msmarco_output_function(data, config, *args, **params):
    if type(data) == dict:
        return binary_output_function(data, config, *args, **params)
    else:
        import bmtrain as bmt
        out_path = config.get("output", "result_path")
        i = 0
        while os.path.exists(os.path.join(out_path, "results_%s_rank_%s.json" % (i, bmt.rank()))):
            i += 1
        fout = open(os.path.join(out_path, "results_%s_rank_%s.json" % (i, bmt.rank())), "w")
        print(json.dumps(data, ensure_ascii=False, indent=2), file=fout)
        fout.close()
        return ""

        qid2pos = json.load(open(config.get("data", "dev_label_path")))
        qid2rank = {}
        for d in data:
            (qid, pid), score = d
            if qid not in qid2rank:
                qid2rank[qid] = []
            qid2rank[qid].append((pid, score))
        mrrscore, total = 0, 0
        for qid in qid2rank:
            if qid not in qid2pos:
                continue
            rank = sorted(qid2rank[qid], key=lambda x:x[1], reverse=True)
            for rankid, p in enumerate(rank[:10]):
                if p[0] in qid2pos[qid]["positive"]:
                    mrrscore += 1 / (rankid + 1)
                    break
            total += 1
        return json.dumps({"MRR@10": round(mrrscore / total, 4)})


def squad_output_function(data, config, *args, **params):
    if data["train"]:
        acc = round(data["right"] / data["total"], 4)
        return json.dumps({"tok_acc": acc})
    else:
        if data['NA_tp'] != 0 or data['NA_fp'] != 0:
            pre = float(data['NA_tp']) / (data['NA_tp'] + data["NA_fp"])
            recall = float(data['NA_tp']) / (data['NA_tp'] + data["NA_fn"])
            if pre + recall == 0:
                naf1 = 0
            else:
                naf1 = 2 * pre * recall / (pre + recall)
        else:
            naf1 = 0

        return json.dumps({
            "EM": round(data["em_sum"] / data["total"], 4),
            "F1": round(data["f1_sum"] / data["total"], 4),
            "NA_F1": round(naf1, 4)
            }
        )

def mlm_output_function(data, config, *args, **params):
    acc = round(data["right"] / data["total"], 4)
    return json.dumps({"tok_acc": acc, "avg_loss": sum(data["loss"]) / len(data["loss"])})

def summarization_output_function(data, config, *args, **params):
    if data["train"]:
        acc = round(data["right"] / data["total"], 4)
        return json.dumps({"tok_acc": acc})
    else:

        return json.dumps({
            "rouge-1": round(data["rouge-1"] / data["total"], 4),
            "rouge-2": round(data["rouge-2"] / data["total"], 4),
            "rouge-l": round(data["rouge-l"] / data["total"], 4)
            }
        )