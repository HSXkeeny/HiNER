class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []
        with open(filename, encoding='utf-8') as f:
            f = f.read()
            for l in tqdm(f.split('\n\n'), desc='Load data'):
                if not l:
                    continue
                sentence, d = [], []
                for i, c in enumerate(l.split('\n')):
                    char, flag = c.split(' ')
                    sentence += char
                    if flag[0] == 'B':
                        d.append([i, i, flag[2:]])
                    elif flag[0] == 'I':
                        d[-1][1] = i
                if len(sentence) > maxlen - 2:
                    continue
                tokens = [tokenizer.tokenize(word)[1:-1] for word in sentence[:maxlen - 2]]
                pieces = [piece for pieces in tokens for piece in pieces]
                tokens_ids = [tokenizer._token_start_id] + tokenizer.tokens_to_ids(pieces) + [tokenizer._token_end_id]
                assert len(tokens_ids) <= maxlen
                length = len(tokens)

                # piece和word的对应关系，中文两者一致，除了[CLS]和[SEP]
                _pieces2word = np.zeros((length, len(tokens_ids)), dtype=np.bool_)
                e_start = 0
                for i, pieces in enumerate(tokens):
                    if len(pieces) == 0:
                        continue
                    pieces = list(range(e_start, e_start + len(pieces)))
                    _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                    e_start += len(pieces)

                # 相对距离
                _dist_inputs = np.zeros((length, length), dtype=np.int64)
                for k in range(length):
                    _dist_inputs[k, :] += k
                    _dist_inputs[:, k] -= k

                for i in range(length):
                    for j in range(length):
                        if _dist_inputs[i, j] < 0:
                            _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                        else:
                            _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
                _dist_inputs[_dist_inputs == 0] = 19

                # golden标签
                _grid_labels = np.zeros((length, length), dtype=np.int64)
                _grid_mask2d = np.ones((length, length), dtype=np.bool_)

                for entity in d:
                    e_start, e_end, e_type = entity[0], entity[1] + 1, entity[-1]
                    if e_end >= maxlen - 2:
                        continue
                    index = list(range(e_start, e_end))
                    for i in range(len(index)):
                        if i + 1 >= len(index):
                            break
                        _grid_labels[index[i], index[i + 1]] = 1
                    _grid_labels[index[-1], index[0]] = categories[e_type]
                _entity_text = set([convert_index_to_text(list(range(e[0], e[1] + 1)), categories[e[-1]]) for e in d])
                D.append((tokens_ids, _pieces2word, _dist_inputs, _grid_labels, _grid_mask2d, _entity_text))
        return D


def collate_fn(data):
    tokens_ids, pieces2word, dist_inputs, grid_labels, grid_mask2d, _entity_text = map(list, zip(*data))

    sent_length = torch.tensor([i.shape[0] for i in pieces2word], dtype=torch.long, device=device)
    # max_wordlen: word长度，非token长度，max_tokenlen：token长度
    max_wordlen = torch.max(sent_length).item()
    max_tokenlen = np.max([len(x) for x in tokens_ids])
    tokens_ids = torch.tensor(sequence_padding(tokens_ids), dtype=torch.long, device=device)
    batch_size = tokens_ids.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = torch.tensor(x, dtype=torch.long, device=device)
        return new_data

    dis_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    dist_inputs = fill(dist_inputs, dis_mat)
    labels_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.long, device=device)
    grid_labels = fill(grid_labels, labels_mat)
    mask2d_mat = torch.zeros((batch_size, max_wordlen, max_wordlen), dtype=torch.bool, device=device)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)
    sub_mat = torch.zeros((batch_size, max_wordlen, max_tokenlen), dtype=torch.bool, device=device)
    pieces2word = fill(pieces2word, sub_mat)

    return [tokens_ids, pieces2word, dist_inputs, sent_length, grid_mask2d], [grid_labels, grid_mask2d, _entity_text]


# 加载数据
train_dataloader = DataLoader(MyDataset(''), ##数据集文件位置
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(MyDataset(''), ##数据集文件位置
                              batch_size=batch_size, collate_fn=collate_fn)
