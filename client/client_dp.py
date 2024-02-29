import argparse
import os

import tensorflow as tf

import flwr as fl


from collections import OrderedDict
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from utils import train_a_model, test_a_model, tranc_floating
from privacy.dp import dp as DiffP
import os
import torch
import copy

# Define Flower client

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model,dataset, trainloaders, valloader, epochs, path,state, device, args, dp, cr):
        self.cid = int(cid)
        # self.model = net(params[0], params[1], params[2])
        self.model = model
        self.dataset=dataset
        # GCN(hidden_channels=32, in_channels=num_node_features, out_channels=num_classes, num_layers=3)
        # self.model=net(hidden_channels=params[1], in_channels=params[0], out_channels=params[2], num_layers=3)
        self.trainloader = trainloaders
        self.valloader = valloader
        self.epochs = epochs
        self.device = device
        self.path = path
        self.state = state
        self.args = args
        self.dp=dp
        self.cr=cr
    def get_parameters(self, config):
        return [val.cpu().numpy() for name, val in self.model.model.state_dict().items() if 'num_batches_tracked' not in name]

    def set_parameters(self, parameters):
        keys = [k for k in self.model.model.state_dict().keys() if 'num_batches_tracked' not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"Training client {self.cid}")
        self.set_parameters(parameters)
        # train(self.model, self.trainloader[self.cid], self.epochs, self.device)
        # params,avg_loss=train(args=self.args, model=self.model, device=self.device, train_graphs=self.trainloader[self.cid], epochs=self.epochs, test_graphs=self.valloader, dp=self.dp)
        # self.set_parameters(params)
        # loss, accuracy = test(self.model, self.trainloader[self.cid], self.device)
        dp_params=[1.1, 0.3]
        test_accuracy=train_a_model(target_model=self.model, dataset=self.dataset, target_indices=self.trainloader, attack_test_indices=self.valloader, num_epochs=self.epochs, batch_size=8, coarsen=self.cr, dp=self.dp, dp_params=dp_params)
        # loss, accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.trainloader[self.cid])
        loss=0
        accuracy=test_a_model(target_model=self.model,dataset=self.dataset, attack_test_indices=self.trainloader)
        try:
            data=pd.read_csv(f"{self.path}/results_train.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data=pd.DataFrame(columns=["Method","Coarsen","Priv","Data","Round", "Client Number", "Loss","Accuracy"])
        data=pd.concat([data, pd.Series(['FL', self.state, self.dp, "Train",config['server_round']-1, self.cid, tranc_floating(loss), tranc_floating(accuracy)], index=data.columns).to_frame().T], ignore_index=True)
        data.to_csv(f"{self.path}/results_train.csv")
        
        # loss, accuracy = test(self.model, self.valloader, self.device)
        # loss,accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.valloader)
        
        return self.get_parameters({}), len(self.trainloader[self.cid]), {}
    def evaluate(self, parameters, config):
        print(f"Evaluating client {self.cid}")
        self.set_parameters(parameters)
        # loss,accuracy = test(self.model, self.valloader, self.device)
        # loss, accuracy = test(args=self.args, model=self.model, device=self.device, test_graphs=self.valloader)
        loss=0
        accuracy=test_a_model(target_model=self.model,dataset=self.dataset, attack_test_indices=self.valloader)
        try:
            os.mkdir(f"{self.path}/clientwise/{self.cid}")
            print('-------FILE CREATED---------')
        except:
            print("")
        #append in file
        with open(f"{self.path}/clientwise/{self.cid}/coarse_{self.state}_{self.dp}.csv", "a") as f:
            f.write(f"{config['server_round']},{tranc_floating(loss)},{tranc_floating(accuracy)}\n")
        try:
            data=pd.read_csv(f"{self.path}/results.csv")
            data.drop(["Unnamed: 0"], axis=1, inplace=True)
        except:
            data=pd.DataFrame(columns=["Method","Coarsen","Priv","Data","Round", "Client Number", "Loss","Accuracy"])
        data=pd.concat([data, pd.Series(['FL', self.state, self.dp, "Test",config['server_round']-1, self.cid, tranc_floating(loss), tranc_floating(accuracy)], index=data.columns).to_frame().T], ignore_index=True)
        data.to_csv(f"{self.path}/results.csv")

        return loss, len(self.valloader), {"accuracy": accuracy}