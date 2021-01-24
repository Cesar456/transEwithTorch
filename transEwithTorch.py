import os
import torch
from torch import nn
from torch.utils import data
import numpy as np

dir_data_path = os.path.join(os.path.dirname(__file__), "./data")


def read_file(data_path, sp="\t"):
    result = {}
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            row = line.strip().split(sp)
            result[row[0]] = int(row[1])
    return result


def open_train(data_path, sp="\t"):
    data_list = []
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            triple = line.strip().split(sp)
            if len(triple) != 3:
                continue
            data_list.append(tuple(triple))
    return data_list


class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, norm=1, dim=100):
        super(TransE, self).__init__()
        self.norm = norm
        self.dim = dim
        self.entities_emb = self._init_emb(entity_count)
        self.relations_emb = self._init_emb(relation_count)

    def _init_emb(self, num_embeddings):
        embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.dim)
        uniform_range = 6 / np.sqrt(self.dim)
        embedding.weight.data.uniform_(-uniform_range, uniform_range)
        embedding.weight.data = torch.div(embedding.weight.data,
                                          embedding.weight.data.norm(p=2, dim=1, keepdim=True))
        return embedding

    def forward(self, positive_triplets: torch.LongTensor, negative_triplets: torch.LongTensor):
        positive_distances = self._distance(positive_triplets)
        negative_distances = self._distance(negative_triplets)
        return positive_distances, negative_distances

    def _distance(self, triplets):
        heads = triplets[:, 0]
        relations = triplets[:, 1]
        tails = triplets[:, 2]
        # L1还是L2距离
        return (self.entities_emb(heads) + self.relations_emb(relations) - self.entities_emb(tails)).norm(p=self.norm, dim=1)


class TripleDataset(data.Dataset):

    def __init__(self, entity2id, relation2id, triple_data_list):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.data = triple_data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        head, tail, relation = self.data[index]
        head_id = self.entity2id[head]
        relation_id = self.relation2id[relation]
        tail_id = self.entity2id[tail]
        return head_id, relation_id, tail_id


if __name__ == '__main__':
    entity_id_dict = read_file(os.path.join(dir_data_path, 'entity2id.txt'))
    relation_id_dict = read_file(os.path.join(dir_data_path, 'relation2id.txt'))
    train_data = open_train(os.path.join(dir_data_path, 'train.txt'))
    valid_data = open_train(os.path.join(dir_data_path, 'valid.txt'))

    # 定义常量
    batch_size = 400
    epochs = 50
    margin = 1.0

    train_dataset = TripleDataset(entity_id_dict, relation_id_dict, train_data)
    valid_dataset = TripleDataset(entity_id_dict, relation_id_dict, valid_data)
    train_data_loader = data.DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = data.DataLoader(valid_dataset, batch_size=batch_size)

    model = TransE(len(entity_id_dict), len(relation_id_dict))
    # adam
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
    print(f"start train")
    for epoch in range(epochs):
        # model.train()
        all_loss = 0
        for i, (local_heads, local_relations, local_tails) in enumerate(train_data_loader):
            positive_triples = torch.stack((local_heads, local_relations, local_tails), dim=1)

            # 生成负样本
            head_or_tail = torch.randint(high=2, size=local_heads.size())
            random_entities = torch.randint(high=len(entity_id_dict), size=local_heads.size())
            broken_heads = torch.where(head_or_tail == 1, random_entities, local_heads)
            broken_tails = torch.where(head_or_tail == 0, random_entities, local_tails)
            negative_triples = torch.stack((broken_heads, local_relations, broken_tails), dim=1)

            optimizer.zero_grad()
            pd, nd = model(positive_triples, negative_triples)
            # pd要尽可能小， nd要尽可能大
            loss = criterion(pd, nd, torch.tensor([-1], dtype=torch.long)).mean()
            loss.backward()
            all_loss += loss.data
            optimizer.step()
            if i % 1000 == 0:
                print(f"epoch:{epoch}/{epochs}, avg_loss={all_loss / (i + 1)}")
        print(f"epoch:{epoch}/{epochs}, all_loss={all_loss}")
