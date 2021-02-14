
from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import shutil
import sys
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertOnlyMLMHead
from tqdm import tqdm
import sys


pretrained_lm = 'bert-base-uncased'
device = 3
model = BertModel.from_pretrained(pretrained_lm,
                                  output_attentions=False,
                                  output_hidden_states=False)
tokenizer = BertTokenizer.from_pretrained(pretrained_lm, do_lower_case=True)
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)
inv_vocab = {k:v for v, k in vocab.items()}


def set_up_dist(rank, world_size):
    dist_port = 12345
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://localhost:{dist_port}',
        world_size=world_size,
        rank=rank
    )
    # create local model
    local_model = model.to(rank)
    local_model = DDP(local_model, device_ids=[rank], find_unused_parameters=True)
    return local_model


def gen_emb_dist(rank, data_dict, world_size, batch_size=16):
    model = set_up_dist(rank, world_size)
    dataset = TensorDataset(data_dict["input_ids"], data_dict["attention_masks"])
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
#     sampler = SequentialSampler(dataset)
    dataset_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)
    model.eval()
    word_emb = defaultdict(list)
    dataset_loader = tqdm(dataset_loader) if rank == 0 else dataset_loader
    
    for batch in dataset_loader:
        with torch.no_grad():
            input_ids = batch[0].to(rank)
            input_mask = batch[1].to(rank)
            outputs = model(input_ids,
                            token_type_ids=None, 
                            attention_mask=input_mask)
            input_mask[:, 0] = 0
            valid_pos = input_mask > 0
            valid_emb = outputs[0][valid_pos]
            valid_id = input_ids[valid_pos]
            for i, token_id in enumerate(valid_id):
                word_emb[token_id.item()].append(valid_emb[i].cpu())
    save_file = os.path.join("../tmp/", f"{rank}_data.pt")
    torch.save(word_emb, save_file)


def gen_emb(data_dict, world_size, batch_size=16):
    if not os.path.exists("../tmp/"):
        os.makedirs("../tmp/")
    mp.spawn(gen_emb_dist, nprocs=world_size, args=(data_dict, world_size, batch_size))
    gather_res = []
    word_emb = defaultdict(list)
    for f in os.listdir("../tmp/"):
        if f[-3:] == '.pt':
            gather_res.append(torch.load(os.path.join("../tmp/", f)))
    assert len(gather_res) == world_size, "Number of saved files not equal to number of processes!"
    for worker_dict in gather_res:
        for word_id, emb in worker_dict.items():
            word_emb[word_id] += emb
    if os.path.exists("../tmp/"):
        shutil.rmtree("../tmp/")
    return word_emb


def load_dataset(dataset_dir, loader_name):
    loader_file = os.path.join(dataset_dir, loader_name)
    assert os.path.exists(loader_file)
    print(f"Loading encoded texts from {loader_file}")
    data = torch.load(loader_file)
    return data


if __name__ == '__main__':
    data = load_dataset("../datasets/nyt/", "text.pt")
    world_size = 4
    print(data["input_ids"][data["attention_masks"] > 0].numel())
    word_emb = gen_emb(data, world_size, batch_size=128)
    save_file = os.path.join("../datasets/nyt/", "word_emb.pt")
    torch.save(word_emb, save_file)



