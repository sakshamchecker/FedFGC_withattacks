import numpy as np

def split_data(dataset, target_ratio, shadow_ratio, attack_train_ratio):
        num_total_graphs = len(dataset)
        num_target_graphs = int(num_total_graphs * target_ratio)
        num_shadow_graphs = int(num_total_graphs * shadow_ratio)
        num_attack_train_graphs = int(num_total_graphs * attack_train_ratio)

        target_indices = np.random.choice(np.arange(num_total_graphs), num_target_graphs, replace=False)
        remain_user_indices = np.setdiff1d(np.arange(num_total_graphs), target_indices)

        shadow_indices = np.random.choice(remain_user_indices, num_shadow_graphs, replace=False)
        remain_user_indices = np.setdiff1d(remain_user_indices, shadow_indices)

        attack_train_indices = np.random.choice(remain_user_indices, num_attack_train_graphs, replace=False)

        attack_test_indices = np.setdiff1d(remain_user_indices, attack_train_indices)
        return target_indices, shadow_indices, attack_train_indices, attack_test_indices
def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.000432batch_size], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))
    returning_prop=np.copy(proportions)
    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size
def create_non_uniform_split(alpha, idxs, client_number, is_train=True):
    N = len(idxs)
    idx_batch_per_client = [[] for _ in range(client_number)]
    (
        idx_batch_per_client,
        min_size,
    ) = partition_class_samples_with_dirichlet_distribution(
        N, alpha, client_number, idx_batch_per_client, idxs
    )
    sample_num_distribution = []

    for client_id in range(client_number):
        sample_num_distribution.append(len(idx_batch_per_client[client_id]))
        

    return idx_batch_per_client
def split_data_to_clients(dataset, num_clients, alpha, target_ratio, shadow_ratio, attack_train_ratio, idxs=None):
    if not idxs:
        idxs=create_non_uniform_split(alpha=alpha, idxs=list(range(len(dataset))), client_number=num_clients, is_train=True)
    # print(len(idxs[0]))
    # print(len(idxs[1]))
    #for all clients, extract the idxs from the dataset and pass it to split data and return the list of target_indices, shadow_indices, attack_train_indices, attack_test_indices
    target_indices, shadow_indices, attack_train_indices, attack_test_indices = [], [], [], []
    for i in range(num_clients):
        target_idx, shadow_idx, attack_train_idx, attack_test_idx = split_data(dataset[idxs[i]], target_ratio, shadow_ratio, attack_train_ratio)
        target_indices.append(target_idx)
        shadow_indices.append(shadow_idx)
        attack_train_indices.append(attack_train_idx)
        attack_test_indices.append(attack_test_idx)
    return target_indices, shadow_indices, attack_train_indices, attack_test_indices, idxs

