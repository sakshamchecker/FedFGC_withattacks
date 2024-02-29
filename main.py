import argparse
import os
import flwr as fl
from datetime import datetime
import numpy as np
from torch_geometric.data import DenseDataLoader
from models.diffpool import DiffPool
from dataset.data_store import DataStore
from dataset.tu_dataset import TUDataset
# from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from utils import train_a_model, test_a_model
from dataset.utils import split_data, split_data_to_clients
from client.client_dp import FlowerClient
from server.server import FedAvgWithAccuracyMetric, FedOptAdamStrategy, FedProxWithAccuracyMetric, fit_config, evaluate_config
import copy
import torch
from attack.attacks import attack_recon, attack_property
torch.manual_seed(42)
np.random.seed(42)
class MyFilter(object):
    def __init__(self, max_nodes):
        self.max_nodes = max_nodes

    def __call__(self, data):
        return data.num_nodes <= self.max_nodes
def load_raw_data( dataset_name,max_nodes=1000, use_feat=True):
        if dataset_name in ['DD', 'PROTEINS']:
            pre_transform = T.Compose([
                T.OneHotDegree(100, cat=False)  # use only node degree as node feature.
            ])
        else:
            pre_transform = T.Compose([
                T.OneHotDegree(10, cat=False)  # use only node degree as node feature.
            ])

        if use_feat:
            dataset = TUDataset("data" + str(max_nodes) + '/', name=dataset_name, use_node_attr=True,
                         use_edge_attr=True, transform=T.ToDense(max_nodes),
                         pre_filter=MyFilter(max_nodes))
        else:
            dataset = TUDataset('data'+ '/'.join((str(max_nodes), 'non_feat')) + '/', name=dataset_name,
                         use_node_attr=True, use_edge_attr=True, transform=T.ToDense(max_nodes),
                         pre_filter=MyFilter(max_nodes), pre_transform=pre_transform)

        if dataset_name in ['OVCAR-8H', 'PC-3', 'MOLT-4H']:
            select_indices = np.random.choice(np.arange(len(dataset)), int(0.1 * len(dataset)))
            dataset = dataset[list(select_indices)]

        if dataset_name == 'PC-3':
            dataset.data.x = dataset.data.x[:, :37]

        if dataset_name == 'OVCAR-8H':
            dataset.data.x = dataset.data.x[:, :65]

        return dataset


def execute(args, cr, dp, experiment_path, attacks):
    # pretrainedvae="pretrained/PROTEINS_PROTEINS_diff_pool_diff_pool_2"
    dataset=load_raw_data("PROTEINS", max_nodes=args.max_nodes)
    target_indices, shadow_indices, attack_train_indices, attack_test_indices = split_data(dataset, args.target_ratio, args.shadow_ratio, args.attack_train_ratio)

    # target_indices, shadow_indices, attack_train_indices, attack_test_indices, idxs = split_data_to_clients(dataset=dataset, num_clients=args.ncl, alpha=args.alpha, target_ratio=args.target_ratio, shadow_ratio=args.shadow_ratio, attack_train_ratio=args.attack_train_ratio)
    # for i in range(len(attack_test_indices)):
    #     print(len(attack_test_indices[i]))
    print("Data Loaded")
    target_model = DiffPool(feat_dim=dataset.num_features, num_classes=dataset.num_classes, max_nodes=20, args=args)
    # print(target_model.model)
    print("Model Loaded")
    test_accuracy=train_a_model(target_model, dataset, target_indices, attack_test_indices, num_epochs=args.epochs, batch_size=8, coarsen=cr, dp=dp, dp_params=[1.1, 0.3])
    train_accuracy=test_a_model(target_model, dataset, attack_test_indices)
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Train Accuracy: {train_accuracy}")
    for att in attacks:
        if att=='recon':
            pretrainedvae="pretrained/PROTEINS_diff_pool.zip"
            attack_recon(target_model, dataset, attack_test_indices, max_nodes=20, recon_stat=['degree_dist', 'close_central_dist', 'between_central_dist','cluster_coeff_dist','isomorphism_test'], recon_metrics=['cosine_similarity'], num_runs=1, graph_vae_model_file=pretrainedvae, experiment_path=experiment_path, cid=-1, cr=cr, dp=dp, round=-1)
        elif att=='infer':
            pretrained_infer="pretrained/PROTEINS_PROTEINS_diff_pool_diff_pool_2"

            attack_property(target_model=target_model, dataset=dataset, attack_test_indices=attack_test_indices, num_runs=3, prop_infer_file=pretrained_infer, properties=['num_nodes', 'num_edges', 'density', 'diameter', 'radius'], path=experiment_path, cid=-1, cr=cr, dp=dp)
    # attack_property(target_model=target_model, dataset=dataset, attack_test_indices=attack_test_indices, num_runs=3, prop_infer_file=pretrainedvae, properties=['num_nodes', 'num_edges', 'density', 'diameter', 'radius'])
    # attack(target_model, dataset, attack_test_indices, max_nodes=20, recon_stat=['degree_dist', 'close_central_dist', 'between_central_dist','cluster_coeff_dist','isomorphism_test'], recon_metrics=['cosine_similarity'], num_runs=1, graph_vae_model_file=pretrainedvae)
# execute(None)

def execute_fl(args, cr, dp, experiment_path, att):
    # experiment_path = f"{args.output}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.ncl}_{args.rounds}_{args.tr_ratio}_{args.epochs}_{args.data}_{args.strat}"
   
    dataset=load_raw_data("PROTEINS", max_nodes=args.max_nodes)
    target_indices, shadow_indices, attack_train_indices, attack_test_indices, idxs = split_data_to_clients(dataset=dataset, num_clients=args.ncl, alpha=args.alpha, target_ratio=args.target_ratio, shadow_ratio=args.shadow_ratio, attack_train_ratio=args.attack_train_ratio)
    #print length  of each val in id
    for i in range(len(attack_test_indices)):
        print(len(attack_test_indices[i]))
    if args.coarsen=='all':
        corasen=[False,True]
    elif args.coarsen=='true':
        corasen=[True]
    else:
        corasen=[False]
    if args.priv=='all':
        priv=[False,True]
    elif args.priv=='true':
        priv=[True]
    else:
        priv=[False]
    # print(type(target_indices))
    # print(type(target_indices[0]))
    # print(len(target_indices))
    # print(len(target_indices[0]))
    # print(len(list(target_indices[0])))
    def client_fn(client_id):
        target_model = DiffPool(feat_dim=dataset.num_features, num_classes=dataset.num_classes, max_nodes=20, args=None)
        # return FlowerClient(cid, net, train_loaders, valloader, args.epochs, path=path, state=coarsen, device=device, args=args, dp=priv)
        return FlowerClient(cid=client_id, model=target_model, dataset=dataset, trainloaders=target_indices[int(client_id)], valloader=attack_test_indices[int(client_id)], epochs=args.epochs, path=experiment_path, state=cr, device="cuda", args=args, dp=dp, cr=cr, attacks=att)
    ray_args = {'num_cpus':1, 'num_gpus':0}
    client_resources = {"num_cpus": 1, "num_gpus": 0}
    if args.strat=="FedAvg":
        st=FedAvgWithAccuracyMetric(
            min_available_clients=int(args.ncl),
            # initial_parameters=get_parameters(net),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )
    elif args.strat=='FedProx':
        st=FedProxWithAccuracyMetric(
            min_available_clients=int(args.ncl),
            # initial_parameters=get_parameters(net),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            proximal_mu=0.1,
        )
    elif args.strat=='FedOptAdam':
        st=FedOptAdamStrategy(
            min_available_clients=int(args.ncl),
            # initial_parameters=get_parameters(net),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )
    history=fl.simulation.start_simulation(
        client_fn=client_fn,
        strategy=st,
        num_clients=int(args.ncl),

        config=fl.server.ServerConfig(num_rounds=int(args.rounds)),
        ray_init_args=ray_args,
        client_resources=client_resources,
    )
# execute(None)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ncl', type=int, default=2, help='number of clients')
    parser.add_argument('--alpha', type=float, default=10, help='alpha')
    parser.add_argument('--target_ratio', type=float, default=0.7, help='target_ratio')
    parser.add_argument('--shadow_ratio', type=float, default=0.0, help='shadow_ratio')
    parser.add_argument('--attack_train_ratio', type=float, default=0.15, help='attack_train_ratio')
    parser.add_argument('--rounds', type=int, default=10, help='rounds')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--strat', type=str, default="FedAvg", help='strategy')
    parser.add_argument('--output', type=str, default="output", help='output')
    parser.add_argument('--coarsen', type=str, default="False", help='coarsen')
    parser.add_argument('--priv', type=str, default="False", help='priv')
    parser.add_argument('--max_nodes', type=int, default=20, help='max_nodes')
    parser.add_argument('--attacks', type=str, default="recon", help='attacks')
    args = parser.parse_args()
    if args.coarsen=='False':
        coarsen=[False]
    elif args.coarsen=='True':
        coarsen=[True]
    else:
        coarsen=[False,True]
    if args.priv=='False':
        priv=[False]
    elif args.priv=='True':
        priv=[True]
    else:
        priv=[False,True]
    if args.attacks=='all':
        attacks=["infer","recon"]
    else:
        attacks=[args.attacks]
    experiment_path=f"{args.output}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.ncl}_{args.rounds}_{args.target_ratio}_{args.epochs}_{args.strat}_{args.coarsen}_{args.priv}"
    print(f"Experiment path: {experiment_path}")
    os.makedirs(experiment_path, exist_ok=True)
    for cr in coarsen:
        for dp in priv:
            print(f"Coarsen: {cr}, Priv: {dp}")
            execute(args, cr, dp, experiment_path, attacks)
    for cr in coarsen:
        for dp in priv:
            print(f"Coarsen: {cr}, Priv: {dp}")
            execute_fl(args, cr, dp, experiment_path, att=attacks)
    # execute(args)