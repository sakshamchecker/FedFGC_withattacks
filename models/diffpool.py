import logging
import os

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.datasets import TUDataset

from models.gnn_base import GNNBase
from models.diffpool_net import DiffPoolNet
from torch_geometric.utils import to_dense_adj

class DiffPool(GNNBase):
    def __init__(self, feat_dim, num_classes, max_nodes, args):
        super(DiffPool, self).__init__(args)

        self.logger = logging.getLogger(__name__)
        self.model = DiffPoolNet(feat_dim, num_classes, max_nodes)
        self.max_nodes=max_nodes

    def train_model(self, train_loader, test_loader, num_epochs=100, dp=False, dp_params=[]):
        self.model.train()
        self.model = self.model.to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_all = 0
        transformer=T.ToDense(self.max_nodes)
        # print(type(train_loader.dataset))
        # train_loader=DenseDataLoader(transformer(train_loader.dataset), batch_size=8, shuffle=True, drop_last=True)
        if dp:
            sigma,clip_value=dp_params[0],dp_params[1]
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")
            # print(len(train_loader[0]))
            for data in train_loader:
                # data = transformer(data)
                # print(type(data))
                data = data.to(self.device)
                
                optimizer.zero_grad()
                print(data.x.shape)
                print(data.adj.shape)
                # print(data.mask.shape)
                # exit()
                output, link_loss, entropy_loss = self.model(data.x, data.adj)
                loss = F.nll_loss(output, data.y.view(-1))
                loss.backward()
                loss_all += data.y.size(0) * loss.item()
                if dp:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)

            # Add noise to gradients
                    for param in self.model.parameters():
                        if param.grad is not None:
                            noise = torch.tensor(torch.randn_like(param.grad) * sigma)
                            param.grad += noise  
                optimizer.step()

            test_acc = self.evaluate_model(test_loader)
            print(f"Test Accuracy: {test_acc:.4f}")
            self.embedding_dim = self.model.graph_embedding.shape[1]

    @torch.no_grad()
    def evaluate_model(self, test_loader):
        self.model.eval()
        self.model = self.model.to(self.device)
        correct = 0

        for data in test_loader:
            data = data.to(self.device)
            pred = self.model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()

        return correct / len(test_loader.dataset)
