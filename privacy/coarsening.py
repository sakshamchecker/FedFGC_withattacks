#first work on SURA 2023

# !pip install deeprobust
# !conda install pytorch torchvision torchaudio -c pytorch
import torch
# print(torch.__version__)
# !pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-{torch.__version__}.html
# !pip install torch-geometric

from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.community import stochastic_block_model
from networkx.generators.random_graphs import watts_strogatz_graph
from networkx.generators.community import random_partition_graph
from torch_geometric.utils import to_dense_adj,to_edge_index, from_scipy_sparse_matrix
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import math
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import FactorAnalysis

import random
import os
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid
import networkx as nx
from networkx.algorithms import community
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.generators.random_graphs import barabasi_albert_graph
from networkx.generators.community import stochastic_block_model
from networkx.generators.random_graphs import watts_strogatz_graph
from networkx.generators.community import random_partition_graph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import math
from tqdm import tqdm
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
import random
from random import sample
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import inv
from torch_geometric.datasets import WebKB
from torch_geometric.utils import to_dense_adj,homophily
import os
import torch
import pickle
import json
import collections
import math
import pandas as pd
import numpy as np
import networkx as nx
# from rdkit import Chem
# from rdkit.Chem import Descriptors
# from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain
# allowable_features = {
#     'possible_atomic_num_list' : list(range(1, 119)),
#     'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
#     'possible_chirality_list' : [
#         Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
#         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
#         Chem.rdchem.ChiralType.CHI_OTHER
#     ],
#     'possible_hybridization_list' : [
#         Chem.rdchem.HybridizationType.S,
#         Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
#         Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
#         Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
#     ],
#     'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
#     'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
#     'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'possible_bonds' : [
#         Chem.rdchem.BondType.SINGLE,
#         Chem.rdchem.BondType.DOUBLE,
#         Chem.rdchem.BondType.TRIPLE,
#         Chem.rdchem.BondType.AROMATIC
#     ],
#     'possible_bond_dirs' : [ # only for double bond stereo information
#         Chem.rdchem.BondDir.NONE,
#         Chem.rdchem.BondDir.ENDUPRIGHT,
#         Chem.rdchem.BondDir.ENDDOWNRIGHT
#     ]
# }

from scipy.sparse import random
from scipy.stats import rv_continuous
from torch_geometric.utils import dense_to_sparse,homophily

# def mol_to_graph_data_obj_simple(mol):
#     """
#     Converts rdkit mol object to graph Data object required by the pytorch
#     geometric package. NB: Uses simplified atom and bond features, and represent
#     as indices
#     :param mol: rdkit mol object
#     :return: graph data object with the attributes: x, edge_index, edge_attr
#     """
#     # atoms
#     num_atom_features = 2   # atom type,  chirality tag
#     atom_features_list = []
#     for atom in mol.GetAtoms():
#         atom_feature = [allowable_features['possible_atomic_num_list'].index(
#             atom.GetAtomicNum())] + [allowable_features[
#             'possible_chirality_list'].index(atom.GetChiralTag())]
#         atom_features_list.append(atom_feature)
#     x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    # num_bond_features = 2   # bond type, bond direction
    # if len(mol.GetBonds()) > 0: # mol has bonds
    #     edges_list = []
    #     edge_features_list = []
    #     for bond in mol.GetBonds():
    #         i = bond.GetBeginAtomIdx()
    #         j = bond.GetEndAtomIdx()
    #         edge_feature = [allowable_features['possible_bonds'].index(
    #             bond.GetBondType())] + [allowable_features[
    #                                         'possible_bond_dirs'].index(
    #             bond.GetBondDir())]
    #         edges_list.append((i, j))
    #         edge_features_list.append(edge_feature)
    #         edges_list.append((j, i))
    #         edge_features_list.append(edge_feature)

    #     # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
    #     edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

    #     # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #     edge_attr = torch.tensor(np.array(edge_features_list),
    #                              dtype=torch.long)
    # else:   # mol has no bonds
    #     edge_index = torch.empty((2, 0), dtype=torch.long)
    #     edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # return data


class CustomDistribution(rv_continuous):
        def _rvs(self,  size=None, random_state=None):
            return random_state.standard_normal(size)
        
def get_laplacian(adj):
    b=torch.ones(adj.shape[0])
    return torch.diag(adj@b)-adj
def convertScipyToTensor(coo):
        try:
            coo = coo.tocoo()
        except:
            coo = coo
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape

        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def experiment(lambda_param,beta_param,alpha_param,gamma_param,C,X_tilde,theta,X,thresh):
      p = X.shape[0]
      k = int(p*0.5)
      n = X.shape[1]
      ones = csr_matrix(np.ones((k,k)))
      ones = convertScipyToTensor(ones)
      ones = ones.to_dense()
      J = np.outer(np.ones(k), np.ones(k))/k
      J = csr_matrix(J)
      J = convertScipyToTensor(J)
      J = J.to_dense()
      zeros = csr_matrix(np.zeros((p,k)))
      zeros = convertScipyToTensor(zeros)
      zeros = zeros.to_dense()
      X_tilde = convertScipyToTensor(X_tilde)
      X_tilde = X_tilde.to_dense()
      C = convertScipyToTensor(C)
      C = C.to_dense()
      eye = torch.eye(k)
      try:
        theta = convertScipyToTensor(theta)
      except:
        theta = theta
      try:
        X = convertScipyToTensor(X)
        X = X.to_dense()
      except:
        X = X

      if(torch.cuda.is_available()):
        # print("yes")
        X_tilde = X_tilde.cuda()
        C = C.cuda()
        theta = theta.cuda()
        X = X.cuda()
        J = J.cuda()
        zeros = zeros.cuda()
        ones = ones.cuda()
        eye = eye.cuda()

      def update(X_tilde,C,i):
          global L
          thetaC = theta@C
          CT = torch.transpose(C,0,1)
          X_tildeT = torch.transpose(X_tilde,0,1)
          CX_tilde = C@X_tilde
          t1 = CT@thetaC + J
          term_bracket = torch.linalg.pinv(t1)
          thetacX_tilde = thetaC@(X_tilde)
          
          L = 1/k

          t1 = -2*gamma_param*(thetaC@term_bracket)
          t2 = alpha_param*(CX_tilde-X)@(X_tildeT)
          t3 = 2*thetacX_tilde@(X_tildeT)
          t4 = lambda_param*(C@ones)
          t5 = 2*beta_param*(thetaC@CT@thetaC)
          T2 = (t1+t2+t3+t4+t5)/L
          Cnew = (C-T2).maximum(zeros)
          t1 = CT@thetaC*(2/alpha_param)
          t2 = CT@C
          t1 = torch.linalg.pinv(t1+t2)
          t1 = t1@CT
          t1 = t1@X
          X_tilde_new = t1
          Cnew[Cnew<thresh] = thresh
          for i in range(len(Cnew)):
              Cnew[i] = Cnew[i]/torch.linalg.norm(Cnew[i],1)
          for i in range(len(X_tilde_new)):
            X_tilde_new[i] = X_tilde_new[i]/torch.linalg.norm(X_tilde_new[i],1)
          return X_tilde_new,Cnew


      for i in (range(20)):
          X_tilde,C = update(X_tilde,C,i)
    
      return X_tilde,C


def coarsening(adj,X,lambda_param,beta_param,gamma_param,alpha_param):
    # adj=to_dense_adj(obj.edge_index)
    # adj=adj[0]
    # edge_list = obj.edge_index
    # NO_OF_EDGES = edge_list.shape[1]
    # X = obj.x
    # X = X.to_dense()
    N = X.shape[0]
    NO_OF_CLASSES = 5


    theta = get_laplacian(adj)
    features = X
    NO_OF_NODES = X.shape[0]
    # NO_OF_CLASSES =  5
    X1=X.type(torch.FloatTensor)
    p = X.shape[0]
    k = int(p*0.5)
    n = X.shape[1]
    # lambda_param = 50
    # beta_param = 50
    # alpha_param = 50
    # gamma_param = 50
    lr = 1e-5
    thresh = 1e-10
    temp = CustomDistribution(seed=1)
    temp2 = temp()  # get a frozen version of the distribution
    X_tilde = random(k, n, density=0.25, random_state=1, data_rvs=temp2.rvs)
    C = random(p, k, density=0.25, random_state=1, data_rvs=temp2.rvs)
    try:
      X2,C2=experiment(lambda_param,beta_param,alpha_param,gamma_param,C,X_tilde,theta,X1,thresh)
    except:
      return [adj,X]
    C_tr=torch.transpose(C2,0,1)
    theta_c=C_tr@theta@C2
    adjtemp = -theta_c
    for i in range(adjtemp.shape[0]):
        adjtemp[i,i]=0
    adjtemp[adjtemp<0.01]=0
    # temp = dense_to_sparse(adjtemp)
    # edge_list_temp = temp[0]
    # number_of_edges = edge_list_temp.shape[1]
    # coo = [[], []]
    # for i in range(len(adjtemp)):
    #     for j in range(len(adjtemp[i])):
    #         # for our purposes, say there is an edge if the value >0
    #         if adjtemp[i][j] >0:
    #             coo[0].append(i)
    #             coo[1].append(j)
    # d = Data(x = X2,edge_index = torch.LongTensor(coo))
    return [adjtemp,X2]
from torch_geometric.data import DenseDataLoader, DataLoader
import scipy.sparse as scpy
from torch_geometric.transforms import ToDense
# from torch_geometric.utils import to_dense


import torch                   



def coarsen_a_data( cus_dataloader, coarsen_params, batch_size):
    """
    Coarsens the data by the given coarsening parameters
    :param cus_dataloader: Custom dataloader object
    :param coarsen_params: Coarsening parameters
    :return: Coarsened data
    """
    # coarsen the data
    training_graphs=[]
    coarsen_adj,coarsen_x=[],[]
    y=[]
    for data in tqdm(cus_dataloader):
        # print(data)
        
        for i in range(len(data)):
            adj=to_dense_adj(data[i].edge_index)[0]
            X=data[i].x
            
            adj1,X1=coarsening(adj,X,coarsen_params[0],coarsen_params[1],coarsen_params[2],coarsen_params[3])
            if(adj1.shape!=adj.shape):
                coarsen_adj.append(adj1.tolist())
                coarsen_x.append(X1.tolist())
            y.append(data.y[i])
            # print(adj1.shape)
            # print(X.shape,X1.shape)
            # A=scpy.csr_matrix(adj1)
            # temp=from_scipy_sparse_matrix(A)
            # g=Data(x=X1,edge_index=temp[0],edge_attr=temp[1],y=graphs[i].y)
            # graphs[i]=g
    coarsen_adj=torch.Tensor(coarsen_adj)
    coarsen_x=torch.Tensor(coarsen_x)
    # print(coarsen_adj,coarsen_x)

    for i in range(len(coarsen_adj)):
        data=Data(x=coarsen_x[i],adj=coarsen_adj[i], y=y[i])
        
    training_graphs=DataLoader(training_graphs, batch_size=batch_size, shuffle=True)
    return training_graphs



    # denser=ToDense(max_nodes)
    # filter=MyFilter(max_nodes)
  # if data.edge_index is not None:
  #         if filter(data):
  #           data=denser(data)
  #       # print(data)

  #       # coarsened_batch=Batch()
  #       # final_batch=coarsened_batch.from_data_list(batch_list)
  #           training_graphs.append(data)
  #         print("corasening")
    