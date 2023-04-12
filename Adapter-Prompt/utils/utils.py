import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn.functional as F
import re
import csv
import json
import uuid
import pickle as pkl
import numpy as np
from copy import deepcopy
import os
import gc ## For garbage collect
from glob import glob
import logging
import pathlib
from collections import OrderedDict
from multiprocessing import Pool
import sys
import time
from utils.settings import args, TASK_DICT, SPECIAL_TOKENS, SPECIAL_TOKEN_IDS, FILL_VAL
from utils.settings import TOKENIZER, LEN_FACTOR, DATA_ATTRS, MEMORY_FACTOR, MODEL_CONFIG, MODEL_CLASS
import quadprog
from gpuutils import GpuUtils
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="UTF-8")
logger = logging.getLogger(__name__)

def make_dir(d):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)

def get_model_dir(tasks):
    return os.path.join(args.model_dir_root, tasks[0]) if args.seq_train_type != "multitask" else args.model_dir_root


def get_losses(parallel_model, cqa, Y, gen_X, gen_Y, loss_fct):
    if "xyz" in args.seq_train_type:
        qa_logits = parallel_model(cqa)
        lm_logits = parallel_model(gen_X)
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
        lm_loss = loss_fct([torch.transpose(l, 1, 2) for l in lm_logits], gen_Y)
        return torch.mean(qa_loss), args.lm_lambda * torch.mean(lm_loss)
    else:
        qa_logits = parallel_model(cqa)
        qa_loss = loss_fct([torch.transpose(l, 1, 2) for l in qa_logits], Y)
        return torch.mean(qa_loss), torch.tensor(0.)
    
class QADataset(Dataset):
    def __init__(self, data_paths, data_type, gen_token, extra_data=[]):
        self.data_type = data_type
        self.gen_token = gen_token
        if args.use_sep:
            self.sep_token = SPECIAL_TOKEN_IDS["sep_token"]
        self.ans_token = SPECIAL_TOKEN_IDS["ans_token"]
        self.eos_token = SPECIAL_TOKEN_IDS["eos_token"]
        self.pad_token = SPECIAL_TOKEN_IDS["pad_token"]

        if not isinstance(data_paths, list):
            data_paths = [data_paths]

        data = []
        for data_path in data_paths:
            if not data_path:
                continue
            with open(data_path, "r") as f:
                raw_ds = json.load(f)
            raw_ds = map(lambda x: x["paragraphs"], raw_ds["data"])
            d = []
            for raw_d in raw_ds:
                d.extend(raw_d)
            data += d
        
        self.data = []
        self.max_a_len = 0
        if len(data_paths)==1 and data_paths[0] is not None and ('wiki' in data_paths[0] or 'woz' in data_paths[0]):
            #data = self._sort_by_index(data)
            #args.n_workers = 1
            if 'wiki' in data_paths[0]:
                answers_file = "wikisql_answers.json" 
            elif 'woz' in data_paths[0]:
                answers_file = "woz.en_answers.json" 
            with open(os.path.join(args.data_dir,answers_file),"r") as f:
                self.answers = json.load(f)
        if len(data) > 0:
            self.data_tokenization(data)


    def etl_single_extra_data(self, data):
        gen_token = data[0]
        data = ' '.join([str(datum) for datum in data[1:]])
        try:
            if args.use_sep:
                context, qa = re.split(str(SPECIAL_TOKEN_IDS["sep_token"]), data)
            else:
                context = ""
                qa = data
            question, answer = re.split(str(SPECIAL_TOKEN_IDS["ans_token"]), qa)
            context = [int(c) for c in context.strip().split()]
            question = [int(q) for q in question.strip().split()]
            answer = [int(a) for a in re.sub(str(SPECIAL_TOKEN_IDS["eos_token"]), "", answer).strip().split()]
            uid = uuid.uuid1().hex
            data = self.parse_example(gen_token, context, question, answer, uid)
        except ValueError:
            return
        return data

    def concat_example(self, gen_token, c, sep_token, q, ans_token, a, eos_token):
        example = sep_token + q + ans_token + a
        if len(example) + 1 > args.max_len:
            logger.warning('an example with len {} is too long!'.format(len(example) + 1))
            return
        example = gen_token + c[:args.max_len-len(example)-1] + example + eos_token
        return example

    def parse_example(self, gen_token, context, question, answer, idx):
        if args.use_sep:
            cq_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [])
        else:
            cq_example = self.concat_example([], context, [], question, [self.ans_token], [], [])
            cqa_example = self.concat_example([], context, [], question, [self.ans_token], answer, [])
        Y_example = self.concat_example([], [], [], [], [], answer, [self.eos_token])
        Y_example = [FILL_VAL] * (len(cqa_example) - len(Y_example)) + Y_example
        if args.use_sep:
            gen_X_example = self.concat_example([gen_token], context, [self.sep_token], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [self.sep_token], question, [self.ans_token], answer, [self.eos_token])
        else:
            gen_X_example = self.concat_example([gen_token], context, [], question, [self.ans_token], answer, [])
            gen_Y_example = self.concat_example([], context, [], question, [self.ans_token], answer, [self.eos_token])
        return cq_example, len(cq_example), cqa_example, len(cqa_example), Y_example, gen_X_example, gen_Y_example, idx

    def parallel_tokenization(self, d):
        examples = []
        context = TOKENIZER.encode(d["context"])
        max_a_len = 0
        for qa in d["qas"]:
            question = TOKENIZER.encode(qa["question"])

            raw_answers = qa["answers"]
            if len(raw_answers) == 0:
                assert qa["is_impossible"]
                raw_answers.append({"text": ""})

            answer = []
            for i, raw_answer in enumerate(raw_answers):
                answer.extend(TOKENIZER.encode(raw_answer["text"]))
                if i != len(raw_answers) - 1:
                    answer.append(self.pad_token)
            max_a_len = max(max_a_len, len(answer))

            examples.append(self.parse_example(self.gen_token, context, question, answer, qa.get("id", 0)))
        return examples, max_a_len

    def data_tokenization(self, data):
        if args.debug:
            data = data[:10]
            new_data = []
            for datum in data:
                new_data.append(self.parallel_tokenization(datum))
            data = new_data
        else:
            with Pool(args.n_workers) as pool:
                data = pool.map(self.parallel_tokenization, data)
        for datum, max_a_len in data:
            self.data.extend(datum)
            self.max_a_len = max(self.max_a_len, max_a_len)

    def sort(self):
        self.data.sort(key=lambda x: len(x[0]))
        return self

    def sort_by_index(self):
        self.data.sort(key=lambda x: x[-1])

    def get_indices(self):
        return [d[-1] for d in self.data]

    #def _sort_by_index(self,data):
    #    datum = []
    #    for d in data:
    #        for qa in d["qas"]:
    #            datum.append({"context":d["context"], "qas":[qa]})
    #    datum.sort(key=lambda x:x["qas"][0]["id"])
    #    return datum

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
class TrainStep:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __call__(self, loss, scheduler_steps):
        if not args.fp32:
            self.optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

        if not args.fp32:
            self.optimizer.update_master_grads()
            self.optimizer.clip_master_grads(args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            
        if args.seq_train_type in args.REG_TYPE_KEYS:
            self.optimizer.step(self.model.reg_params)
        else:
            self.optimizer.step()
        if args.fp32 or (not self.optimizer.overflow):
            for i in range(scheduler_steps):
                self.scheduler.step()
        self.optimizer.zero_grad()
        
class DynamicBatchSampler(Sampler):
    def __init__(self, dataset, data_type, max_batch_size):
        self.dataset = dataset
        self.data_type = data_type
        if data_type == "train":
            self.batch_size = args.train_batch_size
        else:
            self.batch_size = args.test_batch_size
        self.n_samples = len(dataset)
        self.max_batch_size = max_batch_size

    def __iter__(self):
        if args.debug or self.data_type == "test":
            indices = range(self.n_samples)
        else:
            indices = np.random.permutation(self.n_samples)
        max_len, cnt, st = 0, 0, 0
        batch = []
        for ed, idx in enumerate(indices):
            ln = len(self.dataset[idx][2])
            if max(max_len, ln)**LEN_FACTOR * (ed - st + 1) > self.batch_size[cnt]:
                st = ed
                cnt += 1
                max_len = 0
                if cnt == args.n_gpus:
                    yield batch
                    cnt = 0
                    batch = []
            max_len = max(max_len, ln)
            batch.append(idx)
            if len(batch) == self.max_batch_size and self.data_type == "train":
                yield batch
                cnt, max_len, st = 0, 0, ed
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        raise NotImplementedError


def create_dataloader(dataset, data_type, max_batch_size=1000000000):
    if data_type == "train":
        batch_size = args.train_batch_size
    else:
        batch_size = args.test_batch_size

    if isinstance(batch_size, list):
        collate_fn=lambda x,bs=batch_size: dynamic_collate_fn(x, bs)
        shuffle = False
        batch_size = 1
        batch_sampler = DynamicBatchSampler(dataset, data_type, max_batch_size)
    else:
        collate_fn=lambda x: varlen_collate_fn(x)
        shuffle = not (data_type != "train" or args.debug)
        batch_sampler = None

    dataloader =  DataLoader(dataset, num_workers=args.n_workers,
                             collate_fn=collate_fn,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             batch_sampler=batch_sampler)
    return dataloader


class WrapModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapModel, self).__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        return outputs[0]


def remove_id(idx, need_process, all_pasts):
    assert idx in need_process
    del need_process[idx]
    for layer_id in range(MODEL_CONFIG.n_layer):
        all_pasts[layer_id][idx] = 0

def parse_single_real_data(data,task):
    c = data["paragraphs"][0]["context"]
    q = data["paragraphs"][0]["qas"][0]["question"]
    a = data["paragraphs"][0]["qas"][0]["answers"][0]["text"]
    if args.use_sep:
        data = "{}{}{}{}{}{}{}".format(SPECIAL_TOKENS[task],c,SPECIAL_TOKENS["sep_token"],q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    else:
        data = "{}{} {}{}{}{}".format(SPECIAL_TOKENS[task],c,q,SPECIAL_TOKENS["ans_token"],a,SPECIAL_TOKENS["eos_token"])
    return data

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # if top_p > 0.0:
    #     sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #     cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    #     # Remove tokens with cumulative probability above the threshold
    #     sorted_indices_to_remove = cumulative_probs > top_p
    #     # Shift the indices to the right to keep also the first token above the threshold
    #     sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    #     sorted_indices_to_remove[..., 0] = 0

    #     indices_to_remove = sorted_indices[sorted_indices_to_remove]
    #     logits[indices_to_remove] = filter_value
    return logits

def logits_to_tokens(next_logits,test_run):
    filtered_logits = top_k_top_p_filtering(next_logits, top_k=args.top_k_qa, top_p=args.top_p_qa)
    log_probs = F.softmax(filtered_logits, dim=-1)
    next_tokens = torch.multinomial(log_probs, num_samples=1)
    if test_run:
        next_tokens = torch.argmax(log_probs, dim=-1).unsqueeze(-1)  #### [new] Added to fix the constraint output
    return next_tokens

def data_expand(data):
    datum = []
    for d in data:
        para = d["paragraphs"]
        for p in para: 
            for qa in p["qas"]:
                d = {"context": p["context"], "qas": [qa]}
                datum.append({"paragraphs":[d]})
    return datum


def get_split_indices(data_sizes,chunk_sizes):
    ds = deepcopy(data_sizes)
    records = {}
    tmp = {}
    order = 0 # data_sizes index
    i = 0 # chunk_sizes index
    while len(data_sizes)>0:
        d0 = data_sizes[0]
        c0 = chunk_sizes[0]
        if d0>c0:
            val = c0/ds[order]
        else:
            val = d0/ds[order]

        if order not in tmp:
            rec = (0,val)
            tmp[order] = val
        else:
            rec = (tmp[order],tmp[order]+val)
            tmp[order] += val
        if i in records:
            records[i][order] = rec
        else:
            records[i] = {order: rec}

        if d0>c0:
            data_sizes[0]-=c0
            chunk_sizes.pop(0)
            i+=1
        else:
            chunk_sizes[0]-=d0
            data_sizes.pop(0)
            order+=1
            if d0==c0:
                chunk_sizes.pop(0)
                i+=1
    return records


def store_grad(get_ps, grads, grad_dims, task_id): 
    i = 0
    for param in get_ps():
        if param.grad is not None:
            beg = 0 if i == 0 else sum(grad_dims[:i])
            end = sum(grad_dims[:i+1])
            grads[beg: end, task_id].copy_(param.grad.data.view(-1))
        i += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))