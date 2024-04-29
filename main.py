import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.snippets import sequence_padding, Callback, ListDataset, seed_everything
from bert4torch.optimizers import get_linear_schedule_with_warmup
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict, deque
from sklearn.metrics import precision_recall_fscore_support

# 模型参数：训练
epochs = 30  # 训练轮数
steps_per_epoch = None  # 每轮步数
maxlen = 256  # 最大长度
batch_size =32 # 根据gpu显存设置
learning_rate = 1e-3
rorobert_learning_rate = 5e-6
warm_factor = 0.1
weight_decay = 0
use_robert_last_4_layers = True
categories = {'LOC': 2, 'PER': 3, 'ORG': 4} # 根据数据集设置
label_num = len(categories) + 2

# 模型参数：网络结构
lex_emb_size = 200
syn_emb_size = 100
rorobert_hid_size = 768
emb_dropout = 0.5
conv_dropout = 0.5
out_dropout = 0.33

# Rorobert base
config_path = '' # 预训练文件位置
checkpoint_path = ''  # 预训练文件位置
dict_path = ''  # 预训练文件位置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 用到的小函数
def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)

model = Model(use_rorobert).to(device)

class Loss(nn.CrossEntropyLoss):
    def forward(self, outputs, labels):
        grid_labels, grid_mask2d, _ = labels
        grid_mask2d = grid_mask2d.clone()
        return super().forward(outputs[grid_mask2d], grid_labels[grid_mask2d])

rorobert_params = set(model.rorobert.parameters())
other_params = list(set(model.parameters()) - robert_params)
no_decay = ['bias', 'LayerNorm.weight']
params = [
    {'params': [p for n, p in model.rorobert.named_parameters() if not any(nd in n for nd in no_decay)],
     'lr': robert_learning_rate,
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.rrobert.named_parameters() if any(nd in n for nd in no_decay)],
     'lr': robert_learning_rate,
     'weight_decay': 0.0},
    {'params': other_params,
     'lr': learning_rate,
     'weight_decay': weight_decay},
]

optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
updates_total = (len(train_dataloader) if steps_per_epoch is None else steps_per_epoch) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm_factor * updates_total,
                                            num_training_steps=updates_total)
model.compile(loss=Loss(), optimizer=optimizer, scheduler=scheduler, clip_grad_norm=5.0)

#评估
class Evaluator(Callback):

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, p, r, e_f1, e_p, e_r = self.evaluate(valid_dataloader)
        if e_f1 > self.best_val_f1:
            self.best_val_f1 = e_f1
            # model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {p:.5f} r: {r:.5f}')
        print(f'[val-entity level] f1: {e_f1:.5f}, p: {e_p:.5f} r: {e_r:.5f} best_f1: {self.best_val_f1:.5f}\n')

    def evaluate(self, data_loader):
        def cal_f1(c, p, r):
            if r == 0 or p == 0:
                return 0, 0, 0
            r = c / r if r else 0
            p = c / p if p else 0
            if r and p:
                return 2 * p * r / (p + r), p, r
            return 0, p, r

        pred_result = []
        label_result = []

        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        for data_batch in tqdm(data_loader, desc='Evaluate'):
            (token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d), (
            grid_labels, grid_mask2d, entity_text) = data_batch
            outputs = model.predict([token_ids, pieces2word, dist_inputs, sent_length, grid_mask2d])

            grid_mask2d = grid_mask2d.clone()

            outputs = torch.argmax(outputs, -1)
            ent_c, ent_p, ent_r, _ = self.decode(outputs.cpu().numpy(), entity_text, sent_length.cpu().numpy())

            total_ent_r += ent_r
            total_ent_p += ent_p
            total_ent_c += ent_c

            grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
            outputs = outputs[grid_mask2d].contiguous().view(-1)

            label_result.append(grid_labels.cpu())
            pred_result.append(outputs.cpu())

        label_result = torch.cat(label_result)
        pred_result = torch.cat(pred_result)

        p, r, f1, _ = precision_recall_fscore_support(label_result.numpy(), pred_result.numpy(), average="macro")
        e_f1, e_p, e_r = cal_f1(total_ent_c, total_ent_p, total_ent_r)
        return f1, p, r, e_f1, e_p, e_r

    def decode(self, outputs, entities, length):
        class Node:
            def __init__(self):
                self.THW = []  # [(tail, type)]
                self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

        ent_r, ent_p, ent_c = 0, 0, 0
        decode_entities = []
        q = deque()
        for instance, ent_set, l in zip(outputs, entities, length):
            predicts = []
            nodes = [Node() for _ in range(l)]
            count = 0
            for cur in reversed(range(l)):
                # if count >= 29:
                #     print(count)
                count += 1
                heads = []
                for pre in range(cur + 1):
                    # THW
                    if instance[cur, pre] > 1:
                        nodes[pre].THW.append((cur, instance[cur, pre]))
                        heads.append(pre)
                    # NNW
                    if pre < cur and instance[pre, cur] == 1:
                        # cur node
                        for head in heads:
                            nodes[pre].NNW[(head, cur)].add(cur)
                        # post nodes
                        for head, tail in nodes[cur].NNW.keys():
                            if tail >= cur and head <= pre:
                                nodes[pre].NNW[(head, tail)].add(cur)
                # entity
                for tail, type_id in nodes[cur].THW:
                    if cur == tail:
                        predicts.append(([cur], type_id))
                        continue
                    q.clear()
                    q.append([cur])
                    while len(q) > 0:
                        chains = q.pop()
                        for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                            if idx == tail:
                                predicts.append((chains + [idx], type_id))
                            else:
                                q.append(chains + [idx])

            predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
            decode_entities.append([convert_text_to_index(x) for x in predicts])
            ent_r += len(ent_set)
            ent_p += len(predicts)
            ent_c += len(predicts.intersection(ent_set))
        return ent_c, ent_p, ent_r, decode_entities

if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')
