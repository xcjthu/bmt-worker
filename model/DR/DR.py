
import torch
from torch import nn
from model.metric import softmax_acc
import bmtrain as bmt
import os
from model_center.model import Bert

class DenseRetrieval(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(DenseRetrieval, self).__init__()
        self.plmpath = os.path.join(config.get("model", "pretrained_model_path"), config.get("model", "pretrained_model"))

        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        self.que_model = Bert.from_pretrained(self.plmpath)
        self.ctx_model = Bert.from_pretrained(self.plmpath)
        self.loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)
        self.num_train_passage = config.getint("train", "num_train_passage")
        self.rank = bmt.rank()
    
    def gather_tensor(self, tensor):
        all_tensor = bmt.distributed.all_gather(tensor)
        all_tensor[self.rank] = tensor
        return all_tensor

    def forward(self, data, config, gpu_list, acc_result, mode):
        device = data["query_ids"].device
        query_rep = self.que_model(data["query_ids"], attention_mask=data["query_mask"])["last_hidden_state"][:, 0, :] # batch, dim
        ctx_rep = self.ctx_model(data["ctx_ids"], attention_mask=data["ctx_mask"])["last_hidden_state"][:, 0, :] # batch * num_ctx, dim

        all_query = self.gather_tensor(query_rep) # world_size, batch, dim
        all_ctx = self.gather_tensor(ctx_rep) # world_size, batch * num_ctx, dim

        all_query = all_query.view(-1, all_query.shape[-1]) # batch * world_size, dim
        all_ctx = all_ctx.view(-1, all_ctx.shape[-1]) # batch * num_ctx * world_size, dim

        scores = torch.matmul(all_query, all_ctx.transpose(0, 1)) # batch * world_size, batch * num_ctx * world_size
        labels = torch.arange(scores.shape[0]).to(device) * self.num_train_passage # batch * world_size
        
        loss = self.loss_func(scores, labels)
        
        acc_result = softmax_acc(scores, labels, acc_result)
        return {"loss": loss, "acc_result": acc_result}

