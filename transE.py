import os
import random
import numpy as np

dir_data_path = os.path.join(os.path.dirname(__file__), "./data")


def read_file(data_path, sp="\t"):
    with open(data_path) as f:
        lines = f.readlines()
        return [line.strip().split(sp)[0] for line in lines]


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


class TransE:
    def __init__(self, entity_list, relation_list, triple_list, margin=1, learning_rate=1e-2, dim=30, l1=True):
        self.margin = margin
        self.learning_rate = learning_rate
        self.dim = dim
        self.entity_list = entity_list
        self.relation_list = relation_list
        self.triple_list = triple_list
        self.loss = 0
        self.l1 = l1
        self.entity_vector_dict = None
        self.relation_vector_dict = None

    def initialize(self):
        """
        初始化向量
        :return:
        """
        # 随机初始化关系和节点的向量
        entity_vector_list = get_uniform_matrix(self.dim, len(self.entity_list))
        relation_vector_list = get_uniform_matrix(self.dim, len(self.relation_list))
        self.entity_vector_dict = {self.entity_list[i]: entity_vector_list[i] for i in range(len(self.entity_list))}
        self.relation_vector_dict = {self.relation_list[i]: relation_vector_list[i] for i in
                                     range(len(self.relation_list))}

    def train(self, epoch_num=50):
        print("start train")
        n_batches = 400
        batch_size = len(self.triple_list) // n_batches
        print(f"batch size: {batch_size}, epoch_num: {epoch_num}, iter_num: {n_batches * epoch_num}")
        for epoch in range(epoch_num):
            self.loss = 0
            for k in range(n_batches):
                s_batch = random.sample(self.triple_list, batch_size)
                # 元组对（原三元组，打碎的三元组）的列表 ：{((h,r,t),(h',r,t'))}
                t_batch = []
                for row in s_batch:
                    triplet_with_corrupted_triplet = (row, self.get_corrupted_triplet(row))
                    t_batch.append(triplet_with_corrupted_triplet)
                self.update(t_batch)
            print(f"epoch_num:{epoch}, loss:{self.loss}")

    def get_corrupted_triplet(self, row):
        def __get_sample(target_entity):
            while True:
                entity_temp = random.sample(self.entity_list, 1)[0]
                if entity_temp != target_entity:
                    return entity_temp

        # 小于0.5，打坏三元组的第一项
        if random.random() <= 0.5:
            corrupted_triplet = (__get_sample(row[0]), row[1], row[2])
        else:
            corrupted_triplet = (row[0], __get_sample(row[1]), row[2])
        return corrupted_triplet

    def update(self, t_batch):
        for triplet_with_corrupted_triplet in t_batch:
            head_entity_id = triplet_with_corrupted_triplet[0][0]
            tail_entity_id = triplet_with_corrupted_triplet[0][1]
            relation_id = triplet_with_corrupted_triplet[0][2]
            corrupted_head_entity_id = triplet_with_corrupted_triplet[1][0]
            corrupted_tail_entity_id = triplet_with_corrupted_triplet[1][1]
            # tripletWithCorruptedTriplet是原三元组和打碎的三元组的元组tuple
            head_entity_vector = self.entity_vector_dict[head_entity_id]
            tail_entity_vector = self.entity_vector_dict[tail_entity_id]
            relation_vector = self.relation_vector_dict[relation_id]
            corrupted_head_entity_vector = self.entity_vector_dict[corrupted_head_entity_id]
            corrupted_tail_entity_vector = self.entity_vector_dict[corrupted_tail_entity_id]

            # 期望此距离尽可能小
            dist_triplet = distance(head_entity_vector, tail_entity_vector, relation_vector, l1=self.l1)
            # 期望此距离尽可能大
            corrupted_dist_triplet = distance(corrupted_head_entity_vector, corrupted_tail_entity_vector,
                                              relation_vector, l1=self.l1)
            # 合页损失函数
            err = self.margin + dist_triplet - corrupted_dist_triplet
            # 小于0时说明当前向量设置没问题，不需要更新
            if err > 0:
                self.loss += err
                temp_positive = 2 * (tail_entity_vector - head_entity_vector - relation_vector)
                temp_negative = 2 * (corrupted_tail_entity_vector - corrupted_head_entity_vector - relation_vector)
                if self.l1:
                    temp_positive = np.array([1 if s >= 0 else -1 for s in temp_positive])
                    temp_negative = np.array([1 if s >= 0 else -1 for s in temp_negative])
                temp_positive = temp_positive * self.learning_rate
                temp_negative = temp_negative * self.learning_rate

                head_entity_vector += temp_positive
                tail_entity_vector -= temp_positive
                relation_vector += temp_positive - temp_negative
                corrupted_head_entity_vector -= temp_negative
                corrupted_tail_entity_vector += temp_negative

                self.entity_vector_dict[head_entity_id] = head_entity_vector / np.linalg.norm(head_entity_vector)
                self.entity_vector_dict[tail_entity_id] = tail_entity_vector / np.linalg.norm(tail_entity_vector)
                self.relation_vector_dict[relation_id] = relation_vector / np.linalg.norm(relation_vector)
                self.entity_vector_dict[corrupted_head_entity_id] = corrupted_head_entity_vector / np.linalg.norm(
                    corrupted_head_entity_vector)
                self.entity_vector_dict[corrupted_tail_entity_id] = corrupted_tail_entity_vector / np.linalg.norm(
                    corrupted_tail_entity_vector)


def distance(h, t, r, l1=True):
    """
    计算距离
    :param h: 起始节点向量
    :param t: 终止节点向量
    :param r: 关系
    :param l1: 采用L1距离计算，否则采用l2
    :return:
    """
    s = h + r - t
    if l1:
        return np.fabs(s).sum()
    return np.sum(np.square(s))


def get_uniform_matrix(dim, data_size):
    vector_matrix = np.random.uniform(-6 / (dim ** 0.5), 6 / (dim ** 0.5), (data_size, dim))
    return vector_matrix / np.linalg.norm(vector_matrix, axis=1, keepdims=True)


if __name__ == '__main__':
    entity_data = read_file(os.path.join(dir_data_path, 'entity2id.txt'))
    relation_data = read_file(os.path.join(dir_data_path, 'relation2id.txt'))
    triple_data = open_train(os.path.join(dir_data_path, 'train.txt'))
    transE = TransE(entity_data, relation_data, triple_data)
    transE.initialize()
    transE.train()
