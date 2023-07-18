import torch
import numpy as np
import math


class NeibSampler:
    def __init__(self, graph, nb_size, include_self=False):
        """
        nb = neighbors
        """
        n = graph.number_of_nodes()  # 获得图中节点数量
        assert 0 <= min(graph.nodes()) and max(graph.nodes()) < n
        if include_self:  # 如果邻域包含自身
            # 创建邻域权重
            nb_all = torch.zeros(n, nb_size + 1, dtype=torch.int64)
            nb_all[:, 0] = torch.arange(0, n)  # 给每个邻域中心节点加上标号
            nb = nb_all[:, 1:]  # 最终领域矩阵还是不包括自己
        else:  # 如果邻域没有包含自身
            nb_all = torch.zeros(n, nb_size, dtype=torch.int64)
            nb = nb_all
        popkids = []
        for v in range(n):
            nb_v = sorted(graph.neighbors(v))
            # if len(nb_v) == 0:
            #     nb_v.extend([-1] * nb_size)
            #     nb[v] = torch.LongTensor(nb_v)
            # 如果某个节点的邻域数量  小于或者等于  想要的邻域大小
            if len(nb_v) <= nb_size:
                nb_v.extend([-1] * (nb_size - len(nb_v)))  # 使用 -1 进行填充
                nb[v] = torch.LongTensor(nb_v)  # 将邻域信息放入邻域矩阵中
                # 复制N次，确保邻域估计不偏差
                # nb_v = (nb_v * (math.ceil((nb_size - len(nb_v))/len(nb_v))+1))[:nb_size]
                # nb[v] = torch.LongTensor(nb_v)
                # print(nb_v)
                # exit(0)
            else:
                popkids.append(v)
        self.include_self = include_self
        self.g, self.nb_all, self.pk = graph, nb_all, popkids

    def to(self, dev):
        self.nb_all = self.nb_all.to(dev)
        return self

    def sample(self):
        """
        进行邻域采样
        """
        nb = self.nb_all[:, 1:] if self.include_self else self.nb_all  # 分别包不包含自身的情况
        nb_size = nb.size(1)  # 其实就是设定的领域大小，超参中的 nbsz
        pk_nb = np.zeros((len(self.pk), nb_size), dtype=np.int64)  # 过量的领域节点构造成一个二维数组
        # 从这些过量的领域节点中，节点随机采样选取部分邻域作为真正的领域
        for i, v in enumerate(self.pk):
            pk_nb[i] = np.random.choice(sorted(self.g.neighbors(v)), nb_size)
        nb[self.pk] = torch.from_numpy(pk_nb).to(nb.device)
        return self.nb_all
