
from torch_geometric.data import DenseDataLoader, DataLoader
import numpy as np
from privacy.coarsening import coarsen_a_data
def train_a_model(target_model, dataset, target_indices, attack_test_indices, num_epochs, batch_size=32, coarsen=False, dp=False, dp_params=[]):
    # dataset=DataLoader(dataset, batch_size=batch_size)
    # for data in dataset:
    #     graphs=data.to_data_list()
    #     print(graphs[0].edge_index)
    # dataset=coarsen_a_data(cus_dataloader=dataset, coarsen_params=[0.01, 0.01, 0.01, 0.01], batch_size=batch_size)
    target_train_dataset = dataset[list(target_indices)]
    # target_test_dataset = dataset[list(shadow_indices)]
    target_test_dataset = dataset[list(attack_test_indices)]
    target_train_loader = DenseDataLoader(target_train_dataset, batch_size=batch_size)
    target_test_loader = DenseDataLoader(target_test_dataset, batch_size=batch_size)
    if coarsen:
        target_train_loader=coarsen_a_data(cus_dataloader=target_train_loader, coarsen_params=[0.01, 0.01, 0.01, 0.01], batch_size=batch_size)
    target_model.train_model(target_train_loader, target_test_loader, num_epochs, dp, dp_params)
    test_accuracy=target_model.evaluate_model(target_test_loader)
    
    def save_target_model_paras(target_model):
        # target_model.save_model("target_model_parars")
        target_model.save_paras('target_model_parars')
    save_target_model_paras(target_model)
    
    return test_accuracy
 
def test_a_model(target_model,dataset, attack_test_indices):
    target_test_dataset = dataset[list(attack_test_indices)]
    target_test_loader = DenseDataLoader(target_test_dataset, batch_size=8, shuffle=True, drop_last=True)
    test_accuracy=target_model.evaluate_model(target_test_loader)
    return test_accuracy

def tranc_floating(x):
    return float("{:.5f}".format(x))