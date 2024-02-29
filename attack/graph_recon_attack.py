import logging
import math

import torch
import numpy as np
import networkx as nx
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import dgl
import torch
from torch_geometric.data import DenseDataLoader
from torch.utils.data.dataloader import DataLoader
import numpy as np
from models.graphvae import GraphVAE
from torch.utils.data.dataset import Dataset


def to_networkx_adj(adj, mask=None):
    G = nx.Graph()
    
    if mask is not None:
        adj = adj[mask, mask]

    G.add_nodes_from(range(adj.shape[0]))

    for u in range(adj.shape[0]):
        for v in range(u + 1, adj.shape[1]):
            if adj[u, v] == 1:
                G.add_edge(u, v)

    return G



def _to_color(colors):
  colors = colors[colors >= 0]
  self_color = colors.astype(int).astype(str)[0]
  neighbour_color = colors[1:].astype(int).astype(str).tolist()
  return self_color + '|' + ','.join(neighbour_color)




class FineTuneDataset(Dataset):
    def __init__(self, dataset, embedding):
        self.dataset = dataset
        self.embedding = embedding
    
    def __getitem__(self, index):
        return self.dataset[index].adj, self.dataset[index].mask, self.embedding[index]
    
    def __len__(self):
        return len(self.dataset)


def _update_colors(G):
  # G.update_all()
  return list(map(_to_color, G.ndata.pop('color').cpu().numpy()))

def _send_color(edges):
  return {'color': edges.src['color']}


def _gen_create_multiset(num_nodes):
  def _create_multiset(nodes):
    end = nodes.mailbox['color'].shape[1]
    multiset = torch.zeros((nodes.batch_size(), num_nodes)) - 1
    multiset[:, 0] = nodes.data['color']
    multiset[:, 1:end + 1] = nodes.mailbox['color'].sort().values
    return {'color': multiset}
  return _create_multiset
def subtree_kernel(G, H, max_iter=10):
  """Calculate the Weisfeiler Lehman graph kernel.

  Arguments:
      G {networkx.classes.graph.Graph} -- Graph
      H {networkx.classes.graph.Graph} -- Graph

  Keyword Arguments:
      max_iter {int} -- The max number of iterations (default: {10})

  Returns:
      float -- The value of the subtree kernel
  """
  # G = dgl.DGLGraph(G)
  # H = dgl.DGLGraph(H)
  # G = dgl.graph(G)
  # H = dgl.graph(H)
  G = dgl.from_networkx(G)
  H = dgl.from_networkx(H)
  kernel_value = 0

  # Set initial colors
  G.ndata['color'] = torch.zeros(G.number_of_nodes())
  H.ndata['color'] = torch.zeros(H.number_of_nodes())
  N = G.number_of_dst_nodes()
  current_max_color = 0

  # G.register_message_func(_send_color)
  # G.register_reduce_func(_gen_create_multiset(N))
  # H.register_message_func(_send_color)
  # H.register_reduce_func(_gen_create_multiset(N))
  G.update_all(_send_color, _gen_create_multiset(N))
  H.update_all(_send_color, _gen_create_multiset(N))

  # Refine colors until convergence
  for i in range(max_iter):
    G_colors = _update_colors(G)
    H_colors = _update_colors(H)

    G_unique_colors, G_counts = np.unique(G_colors, return_counts=True)
    H_unique_colors, H_counts = np.unique(H_colors, return_counts=True)
    G_multiset = dict(zip(G_unique_colors, G_counts))
    H_multiset = dict(zip(H_unique_colors, H_counts))

    # Recoloring (str -> int)
    unique_colors = np.unique(np.append(G_unique_colors, H_unique_colors))
    recolor_map = {color: i + 1 for i, color in enumerate(unique_colors)}

    # Add the value of the subtree kernel in i-th step
    G_color_vector = np.zeros(len(unique_colors))
    H_color_vector = np.zeros(len(unique_colors))
    for color, i in recolor_map.items():
      G_color_vector[i - 1] = G_multiset[color] if color in G_multiset else 0
      H_color_vector[i - 1] = H_multiset[color] if color in H_multiset else 0

    kernel_value += np.dot(G_color_vector, H_color_vector)

    temp = np.array([recolor_map[color] for color in G_colors]) + current_max_color
    G.ndata['color'] = torch.from_numpy(np.array([recolor_map[color] for color in G_colors]) + current_max_color)
    H.ndata['color'] = torch.from_numpy(np.array([recolor_map[color] for color in H_colors]) + current_max_color)
    current_max_color += len(unique_colors)

  return kernel_value


class Attack:
    def __init__(self, target_model, max_nodes):
        self.logger = logging.getLogger('attack')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model.to(self.device)
        self.max_nodes = max_nodes

    def determine_stat(self, graph_recon_stat):
        self.graph_recon_stat =graph_recon_stat

        if graph_recon_stat == 'degree_dist':
            self.stat = self.degree_dist_error
        elif graph_recon_stat == 'cluster_coeff_dist':
            self.stat = self.cluster_coeff_dist_error
        elif graph_recon_stat == 'between_central_dist':
            self.stat = self.between_central_dist_error
        elif graph_recon_stat == 'close_central_dist':
            self.stat = self.close_central_dist_error
        elif graph_recon_stat == 'isomorphism_test':
            self.stat = self.isomorphism_test
        else:
            raise Exception('unsupported graph recon stat')

    def determine_metric(self, graph_recon_metric):
        self.graph_recon_metric = graph_recon_metric

        if graph_recon_metric == 'l2':
            self.metric = self._l2_dist
        elif graph_recon_metric == 'cosine_similarity':
            self.metric = self._cosine_similarity
        elif graph_recon_metric == 'wasserstein':
            self.metric = self._wasserstein
        elif graph_recon_metric == 'kl':
            self.metric = self._kl
        elif graph_recon_metric == 'jsd':
            self.metric = self._jsd
        else:
            raise Exception('unsupported graph recon metric')

    def evaluate_reconstruction(self, test_dataset, recon_adjs):
        self.logger.info('evaluating reconstruction')

        errors = np.zeros(len(test_dataset))
        ori_dist_list, recon_dist_list = [], []
        ori_adj_list, recon_adj_list = [], []
        ori_stat_list, recon_stat_list = [], []
        for i, graph in enumerate(test_dataset):
            ori_adj = graph.adj.numpy()

            if isinstance(recon_adjs[i], torch.Tensor):
                recon_adj = recon_adjs[i].numpy()
            else:
                recon_adj = recon_adjs[i]

            if self.graph_recon_stat == 'isomorphism_test':
                errors[i] = self.stat(ori_adj, recon_adj)
            else:
                ori_stat, recon_stat, ori_dist, recon_dist = self.stat(ori_adj, recon_adj)
                ori_stat_list.append(ori_stat)
                ori_dist_list.append(ori_dist)
                ori_adj_list.append(ori_adj)
                recon_stat_list.append(recon_stat)
                recon_dist_list.append(recon_dist)
                recon_adj_list.append(recon_adj)
                errors[i] = self.metric(ori_dist, recon_dist)

        # # visualize the closest graphs
        # if self.graph_recon_stat is not 'isomorphism_test':
        #     if self.graph_recon_metric in ['cosine_similarity']:
        #         select_index = np.argmax(errors)
        #     else:
        #         select_index = np.argmin(errors)
        #     self.visualization_stat(ori_stat_list[select_index], recon_stat_list[select_index])
        #     self.visualization_dist(ori_dist_list[select_index], recon_dist_list[select_index])

        #     # visualize the whole dataset by node.
        #     self.visualization_stat(np.asarray(ori_stat_list).reshape(1, -1)[0], np.asarray(recon_stat_list).reshape(1, -1)[0],
        #                             out="full_" + str(self.graph_recon_stat) + ".pdf")

        # # visualize the graphs
        # if False:
        #     for i in range(len(errors)):
        #         self.visualize_adj(ori_adj_list[i], recon_adj_list[i], save_name=str(i))

        # return np.mean(errors)
        if np.isinf(errors).any():
            return np.ma.masked_invalid(errors).mean()
        else:
            return np.mean(errors) # np.ma.masked_invalid(errors).mean() # use mask if there is any inf values.

    def adj_to_graph(self, adj):
        rows, cols = np.where(adj == 1)
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        return gr

    def visualize_adj(self, adj1, adj2, save_name):
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        gr1 = self.adj_to_graph(adj1)
        gr2 = self.adj_to_graph(adj2)
        nx.draw(gr1, node_size=500, with_labels=True, ax=axes[0])
        nx.draw(gr2, node_size=500, with_labels=True, ax=axes[1])
        plt.show()
        plt.savefig(config.ATTACK_DATA_PATH+"/graph_reconstruct/visualization/"+save_name+".pdf")
        a = 1

    def isomorphism_test(self, ori_adj, recon_adj):
        nx_ori_graph = to_networkx_adj(ori_adj)
        nx_recon_graph = to_networkx_adj(recon_adj)

        # wl = GK_WL()
        # kernel_value = wl.compare(nx_ori_graph, nx_recon_adj, h=10, node_label=False)
        kernel_value = subtree_kernel(nx_ori_graph, nx_recon_graph, max_iter=10)
        ori_kernel_value = subtree_kernel(nx_ori_graph, nx_ori_graph, max_iter=10)
        recon_kernel_value = subtree_kernel(nx_recon_graph, nx_recon_graph, max_iter=10)
        norm_kernel_value = kernel_value / math.sqrt(ori_kernel_value * recon_kernel_value)

        return norm_kernel_value

    def visualization_stat(self, true_stat, est_stat, out=None):
        # import seaborn as sns
        # out = "_".join((self.graph_recon_stat, self.graph_recon_metric))
        # sns.displot([true_stat, est_stat], col_order=['r', 'b'], alpha=0.6)
        # plt.show()
        plt.style.use('seaborn-darkgrid')
        plt.rcParams['text.usetex'] = False
        # plt.style.use(['no-latex'])
        FONTSIZE = 35

        title = {
            "degree_dist": "Degree Distribution",
            "close_central_dist": "CC Distribution",
            "between_central_dist": "BC Distribution",
            "cluster_coeff_dist": "LCC Distribution"
        }
        from matplotlib.ticker import MaxNLocator
        fig = plt.figure(figsize=(9, 5))  # Creates a new figure
        fig.suptitle(str(title[self.graph_recon_stat]), fontsize=FONTSIZE)  # Add the text/suptitle to figure
        ax = fig.add_subplot(111)
        if self.graph_recon_stat == "degree_dist":
            ax.hist([true_stat, est_stat], bins=np.linspace(0, 20, 20), alpha=0.7,
                    label=["Target Graph", "Reconstructed Graph"])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.legend(fontsize=FONTSIZE - 5)
        else:
            ax.hist([true_stat, est_stat], bins=np.linspace(0.0, 1.0, 10), alpha=0.7,
                    label=["Target Graph", "Reconstructed Graph"])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.xlabel('Bucksize', size=30)
        plt.ylabel('Number of Nodes', size=FONTSIZE)
        plt.xticks(size=FONTSIZE-5)
        plt.yticks(size=FONTSIZE-5)
        plt.tight_layout()
        if out is None:
            out = title[self.graph_recon_stat]
        plt.savefig(config.PLOT_PATH + out + ".pdf", bbox_inches="tight")
        plt.show()

        a = 1

    def visualization_dist(self, true_dist, est_dist):
        # for graph_recon_stat in self.args['graph_recon_stat']:
        #     for graph_recon_metric in self.args['graph_recon_metric']:
        #         for i in list(self.graph_recon_dist[graph_recon_stat][graph_recon_metric].keys()):
        #             x = self.graph_recon_dist[graph_recon_stat][graph_recon_metric][i][0]
        #             y = self.graph_recon_dist[graph_recon_stat][graph_recon_metric][i][1]
        self.scatter_plot_2d(true_dist, est_dist, save_name="_".join((self.graph_recon_stat, self.graph_recon_metric)))

    def degree_dist_error(self, ori_adj, recon_adj):
        ori_dist = self._cal_degree_dist(ori_adj)
        ori = self._cal_degree(ori_adj)
        recon_dist = self._cal_degree_dist(recon_adj)
        recon = self._cal_degree(recon_adj)
        return ori, recon, ori_dist, recon_dist

    def cluster_coeff_dist_error(self, ori_adj, recon_adj):
        ori_dist = self._cal_cluster_coeff_dist(ori_adj)
        ori = self._cal_cluster_coeff(ori_adj)
        recon_dist = self._cal_cluster_coeff_dist(recon_adj)
        recon = self._cal_cluster_coeff(recon_adj)
        return ori, recon, ori_dist, recon_dist

    def between_central_dist_error(self, ori_adj, recon_adj):
        ori_dist = self._cal_between_central_dist(ori_adj)
        ori= self._cal_between_central(ori_adj)
        recon_dist = self._cal_between_central_dist(recon_adj)
        recon = self._cal_between_central(recon_adj)
        return ori, recon, ori_dist, recon_dist

    def close_central_dist_error(self, ori_adj, recon_adj):
        ori_dist = self._cal_close_central_dist(ori_adj)
        ori = self._cal_close_central(ori_adj)
        recon_dist = self._cal_close_central_dist(recon_adj)
        recon = self._cal_close_central(recon_adj)
        return ori, recon, ori_dist, recon_dist

    def _cal_degree_dist(self, adj):
        degree = np.sum(adj, axis=1).astype(np.int64)
        degree_dist = np.bincount(degree, minlength=self.max_nodes)
        return degree_dist / np.sum(degree_dist)

    def _cal_cluster_coeff_dist(self, adj):
        nx_graph = to_networkx_adj(adj)
        cluster_coeff = nx.algorithms.cluster.clustering(nx_graph)
        cluster_coeff = np.array(list(cluster_coeff.values()))
        cluster_coeff_dist = np.histogram(cluster_coeff, bins=10, range=(0.0, 1.0))
        return cluster_coeff_dist[0] / np.sum(cluster_coeff_dist[0])

    def _cal_degree(self, adj):
        degree = np.sum(adj, axis=1).astype(np.int64)
        degree_dist = np.bincount(degree, minlength=self.max_nodes)
        return degree

    def _cal_cluster_coeff(self, adj):
        nx_graph = to_networkx_adj(adj)
        cluster_coeff = nx.algorithms.cluster.clustering(nx_graph)
        cluster_coeff = np.array(list(cluster_coeff.values()))
        return cluster_coeff

    def _cal_between_central(self, adj):
        nx_graph = to_networkx_adj(adj)
        between_central = nx.algorithms.centrality.betweenness_centrality(nx_graph)
        between_central = np.array(list(between_central.values()))
        return between_central

    def _cal_close_central(self, adj):
        nx_graph = to_networkx_adj(adj)
        close_central = nx.algorithms.centrality.closeness_centrality(nx_graph)
        close_central = np.array(list(close_central.values()))
        return close_central

    def _cal_between_central_dist(self, adj):
        nx_graph = to_networkx_adj(adj)
        between_central = nx.algorithms.centrality.betweenness_centrality(nx_graph)
        between_central = np.array(list(between_central.values()))
        between_central_dist = np.histogram(between_central, bins=10, range=(0.0, 1.0))
        return between_central_dist[0] / np.sum(between_central_dist[0])

    def _cal_close_central_dist(self, adj):
        nx_graph = to_networkx_adj(adj)
        close_central = nx.algorithms.centrality.closeness_centrality(nx_graph)
        close_central = np.array(list(close_central.values()))
        close_central_dist = np.histogram(close_central, bins=10, range=(0.0, 1.0))
        return close_central_dist[0] / np.sum(close_central_dist[0])

    def _l2_dist(self, true_dist, est_dist):
        return np.sqrt(np.sum((true_dist - est_dist) ** 2))

    def _kl(self, true_dist, est_dist):
        from scipy.stats import entropy
        # out = F.kl_div(true_dist, est_dist) # need the input to be tensor.
        # out = sum(true_dist * np.log(true_dist / est_dist))  # will raise error when est_dist is zero values.
        out = entropy(true_dist, est_dist) # note the inf value when est_dist is zero values.
        return out

    def _jsd(self, true_dist, est_dist):
        js_distance = jensenshannon(true_dist, est_dist)
        return js_distance ** 2

    def _wasserstein(self, true_dist, est_dist):
        values = np.linspace(0.1, 1.0, len(true_dist))
        # return wasserstein_distance(true_dist, est_dist)
        return wasserstein_distance(values, values, true_dist, est_dist)

    def _cosine_similarity(self, true_dist, est_dist):
        return cosine_similarity(true_dist.reshape(1, -1), est_dist.reshape(1, -1))

    def scatter_plot_2d(self, x, y, save_name, y_lim_max=1):
        import matplotlib.pyplot as plt
        import seaborn
        seaborn.set_style('darkgrid', {'legend.frameon': True})
        plt.rcParams['text.usetex'] = True
        kwargs = dict(alpha=0.5, bins=20)
        x_limit=6
        s = np.arange(0, x_limit)
        plt.bar(s, x[:x_limit], alpha=0.5, color='r', label='Original Graph')
        plt.bar(s, y[:x_limit], alpha=0.5, color='b', label='Reconstructed Graph')
        plt.gca().set(title='Original/Reconstructed Graph distribution', ylabel='Distribution')
        plt.xlabel("Node ID")
        plt.ylim(0, y_lim_max)
        # plt.annotate("ACC:%.3f AUC:%.3f" % (acc, auc), xy=(x_lim_max / 2, 50))
        plt.legend()
        plt.tight_layout()
        plt.savefig(config.PLOT_PATH + save_name + '.pdf')
        plt.show()

        a=1




class AttackGraphRecon(Attack):
    def __init__(self, target_model, max_nodes):
        super(AttackGraphRecon, self).__init__(target_model, max_nodes)

    def init_graph_vae(self, dataset, embedding_dim, max_nodes):
        self.graph_vae = GraphVAE(dataset.num_features, embedding_dim,
                                  dataset.num_classes, max_nodes)
        # print(self.graph_vae.model)
        # exit()
    def train_gae(self, train_dataset, num_epoch):
        self.logger.info('training gae model')
        
        train_loader = DenseDataLoader(train_dataset, batch_size=1)
        self.graph_vae.train_model(train_loader, num_epoch=num_epoch)
        
    def gen_fine_tune_dataset(self, train_dataset, embedding_dim):
        self.logger.info('generating embedding')
        graph_embedding = torch.zeros([len(train_dataset), embedding_dim], dtype=torch.float32)
    
        for i, data in enumerate(train_dataset):
            x, adj, mask = self._gen_input_data(data)
            self.target_model(x, adj, mask)
            graph_embedding[i] = self.target_model.graph_embedding.cpu().detach().float()

        self.fine_tune_dataset = FineTuneDataset(train_dataset, graph_embedding)
        
    def fine_tune_gae(self, num_epoch):
        self.logger.info('fine tuning gae')
        
        fine_tune_loader = DataLoader(self.fine_tune_dataset, batch_size=1)
        self.graph_vae.fine_tune_model(fine_tune_loader, num_epoch=num_epoch)

    def gen_test_embedding(self, test_dataset):
        self.logger.info('generating test embedding')

        test_graph_embedding = []
        for data in test_dataset:
            x, adj, mask = self._gen_input_data(data)
            self.target_model(x, adj, mask)
            test_graph_embedding.append(self.target_model.graph_embedding)

        self.test_graph_embedding = torch.stack(test_graph_embedding)

    def reconstruct_graph(self):
        self.logger.info('reconstructing graph')
    
        self.recon_adjs = []
        for graph_embedding in self.test_graph_embedding:
            self.recon_adjs.append(self.graph_vae.model.reconstruct(graph_embedding))

    def save_model(self, save_path):
        self.graph_vae.save_model(save_path)

    def load_model(self, save_path):
        self.graph_vae.load_model(save_path)

    def save_data(self, save_path):
        torch.save(self.recon_adjs, save_path)

    def load_data(self, save_path):
        self.recon_adjs = torch.load(save_path)

    def _gen_input_data(self, graph):
        x = graph.x.reshape([1, graph.x.shape[0], graph.x.shape[1]]).to(self.device)
        adj = graph.adj.reshape([1, graph.adj.shape[0], graph.adj.shape[1]]).to(self.device)
        mask = graph.mask.reshape([1, graph.mask.shape[0]]).to(self.device)

        return x, adj, mask