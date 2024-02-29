from attack.graph_recon_attack import AttackGraphRecon
from collections import defaultdict
from attack.property_infer import Attack
import numpy as np
import os
def attack_recon(target_model,dataset, attack_test_indices, max_nodes, recon_stat, recon_metrics, num_runs, graph_vae_model_file, experiment_path, cr, dp, cid, round):
    attack = AttackGraphRecon(target_model.model, max_nodes)
    # paras=target_model.paras
    paras = target_model.load_paras('target_model_parars')
    attack.init_graph_vae(dataset, 192, max_nodes)
    # print(attack.graph_vae_model)
    graph_recon_stat_run = []
    graph_recon_stat = defaultdict(dict)
    graph_recon_dist = defaultdict(dict)
    attack_test_dataset = dataset[list(attack_test_indices)]
    attack.load_model(graph_vae_model_file)
    if len(attack_test_dataset) > 20:
        select = np.random.choice(np.arange(len(attack_test_dataset)), 2)
        attack_test_dataset = attack_test_dataset[select.tolist()]
        attack.gen_test_embedding(attack_test_dataset)
        attack.reconstruct_graph()
# '''
# save recon data from datastore
# '''
    graph_recon_stat_data = defaultdict(dict)
    for graph_recon_stat in recon_stat:
        attack.determine_stat(graph_recon_stat)
        for graph_recon_metric in recon_metrics:
            attack.determine_metric(graph_recon_metric)

            metric_value = attack.evaluate_reconstruction(attack_test_dataset, attack.recon_adjs)
            graph_recon_stat_data[graph_recon_stat][graph_recon_metric] = metric_value

    graph_recon_stat_run.append(graph_recon_stat_data)

    for graph_recon_stat in recon_stat:
        for graph_recon_metric in recon_metrics:

            run_data = np.zeros(num_runs)
            for run in range(num_runs):
                run_data[run] = graph_recon_stat_run[run][graph_recon_stat][graph_recon_metric]

            metric_value = [np.mean(run_data), np.std(run_data)]
            graph_recon_stat_data[graph_recon_stat][graph_recon_metric] = metric_value

            print('graph_recon_stat: %s, graph_recon_metric: %s, %s' %
                                (graph_recon_stat, graph_recon_metric, metric_value))
    #save to file
    try:
        os.mkdir(f"{experiment_path}/recon")
    except:
        print("")
    with open(f"{experiment_path}/recon/recon_results.csv", "a+") as f:
        f.write(f"cid,round, cr, dp, Graph Recon Stat, Graph Recon Metric, Value, Std\n")
        for graph_recon_stat in recon_stat:
            for graph_recon_metric in recon_metrics:
                f.write(f"{cid},{round},{cr},{dp},{graph_recon_stat}, {graph_recon_metric}, {graph_recon_stat_data[graph_recon_stat][graph_recon_metric][0]}, {graph_recon_stat_data[graph_recon_stat][graph_recon_metric][1]}\n")
def attack_property(target_model, dataset,attack_train_indices, attack_test_indices,num_runs, prop_infer_file, properties, path, cid, cr, dp):
    acc_run=[]
    baseline_acc_run=[]
    acc={}
    baseline_acc={}
    attack=Attack(target_model=target_model.model, shadow_model=target_model.model, properties=properties, property_num_class=2)
    attack_test_dataset = dataset[list(attack_test_indices)]
    attack_train_dataset = dataset[list(attack_train_indices)]
    
    # attack.generate_labels(attack_test_dataset, attack_test_dataset, property_num_class)
    # print("generated test embedding")
    
    # attack.load_data(prop_data_file)
    attack.generate_test_embedding(attack_test_dataset,192)
    attack.generate_train_embedding(attack_train_dataset,192)
    attack.generate_labels(attack_train_dataset, attack_test_dataset, 2)
    # print("generated labels")
    # attack.train_attack_model(is_train=False)
    
    
    # print("trained attack model")
    # attack.load_attack_model(prop_infer_file)
    # print("loaded attack model")

    attack.train_attack_model()
    
    for i in range(num_runs):
        acc_run.append(attack.evaluate_attack_model())
        baseline_acc_run.append(attack.baseline_acc)
    for property in properties:
            run_data = np.zeros(num_runs)
            baseline_run_data = np.zeros(num_runs)

            for run in range(num_runs):
                run_data[run] = acc_run[run][property]
                baseline_run_data[run] = baseline_acc_run[run][property]

            acc[property] = [np.mean(run_data), np.std(run_data)]
            baseline_acc[property] = [np.mean(baseline_run_data), np.std(baseline_run_data)]
    print(acc)
    print(baseline_acc)
    try:
        os.mkdir(f"{path}/attack")
    except:
        print("")
    with open(f"{path}/attack/attack_results.csv", "a+") as f:
        f.write(f"cid, cr, dp, Property, Accuracy, Baseline Accuracy\n")
        for property in properties:
            f.write(f"{cid},{cr},{dp},{property}, {acc[property][0]}, {baseline_acc[property][0]}\n")