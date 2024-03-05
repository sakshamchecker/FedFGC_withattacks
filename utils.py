
from torch_geometric.data import DenseDataLoader, DataLoader
import numpy as np
from torch_geometric.transforms import ToDense
from privacy.coarsening import coarsen_a_data
from torch_geometric.data import Batch
class MyFilter(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes
    
    def __call__(self, data):
        return data.num_nodes <= self.max_nodes
    
def train_a_model(target_model, dataset, target_indices, attack_test_indices, num_epochs, batch_size=32, coarsen=False, dp=False, dp_params=[], max_nodes=20):
    # dataset=DataLoader(dataset, batch_size=batch_size)
    # for data in dataset:
    #     graphs=data.to_data_list()
    #     print(graphs[0].edge_index)
    # dataset=coarsen_a_data(cus_dataloader=dataset, coarsen_params=[0.01, 0.01, 0.01, 0.01], batch_size=batch_size)
    target_train_dataset = dataset[list(target_indices)]
    # target_test_dataset = dataset[list(shadow_indices)]
    target_test_dataset = dataset[list(attack_test_indices)]
    target_train_loader = DataLoader(target_train_dataset, batch_size=batch_size)
    target_test_loader = DataLoader(target_test_dataset, batch_size=batch_size)
    if coarsen:
        target_train_loader=coarsen_a_data(cus_dataloader=target_train_loader, coarsen_params=[0.01, 0.01, 0.01, 0.01], batch_size=batch_size)
    filtered_train_loader=[]
    filtered_test_loader=[]
    denser=ToDense(max_nodes)
    filter=MyFilter(max_nodes)
    # for data in target_train_loader:
    #     g=data.to_data_list()
    #     for i in range(len(data)):
    #         if filter(data[i]):
    #             temp=denser(data[i])
    #             filtered_train_loader.append(temp)
    #             # g.append(temp)
    #             print(temp.x.shape)
    #             print(temp.adj.shape)
    #             print(temp.mask.shape)
    #         # filtered_train_loader.append(Batch().from_data_list(g))
    # print('------------------------------', len(filtered_train_loader))
    for data in target_test_loader:
        for i in range(len(data)):
            if filter(data[i]):
                temp=denser(data[i])
                filtered_test_loader.append(temp)
    # target_train_loader=DenseDataLoader(filtered_train_loader, batch_size=batch_size, drop_last=True)
    target_test_loader=DenseDataLoader(filtered_test_loader, batch_size=batch_size, drop_last=True)
    target_model.train_model(target_train_loader, target_test_loader, num_epochs, dp, dp_params)
    test_accuracy=target_model.evaluate_model(target_test_loader)
    
    def save_target_model_paras(target_model):
        # target_model.save_model("target_model_parars")
        target_model.save_paras('target_model_parars')
    save_target_model_paras(target_model)
    
    return test_accuracy
 
def test_a_model(target_model,dataset, attack_test_indices):
    target_test_dataset = dataset[list(attack_test_indices)]
    filter=MyFilter(20)
    denser=ToDense(20)
    target_test_loader = DataLoader(target_test_dataset, batch_size=8, shuffle=True, drop_last=True)
    filtered_test_loader=[]
    for data in target_test_loader:
        for i in range(len(data)):
            if filter(data[i]):
                temp=denser(data[i])
                filtered_test_loader.append(temp)
    target_test_loader=DenseDataLoader(filtered_test_loader, batch_size=8, drop_last=True)
    test_accuracy=target_model.evaluate_model(target_test_loader)
    return test_accuracy

def tranc_floating(x):
    return float("{:.5f}".format(x))