
import logging
import os
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN

# Data manipulation
import pandas as pd  # for data manipulation
import numpy as np  # for data manipulation
import torch
# Visualization
import plotly.express as px  # for data visualization
import matplotlib.pyplot as plt  # for showing handwritten digits

# UMAP dimensionality reduction
from umap import UMAP

import torch
from torch.linalg import cholesky
import numpy as np
from scipy.linalg import logm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
logger = logging.getLogger(__name__)


#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    gpu_ids=[0]
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model








np.random.seed(42)
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
    '''
    # 总类别数
    # print(len(train_labels))#60000
    n_classes = train_labels.max() + 1  # 总类别数目

    # [alpha]*n_clients 如下：
    # [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # 得到 62 * 10 的标签分布矩阵，记录每个 client 占有每个类别的比率
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    # print(label_distribution)

    # 记录每个类别对应的样本下标
    # 返回二维数组
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 定义一个空列表作最后的返回值
    client_idcs = [[] for _ in range(n_clients)]

    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs





#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def create_datasets(data_path, dataset_name, num_clients, num_shards, iid,alpha):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    # dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),#转换为tensor
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#归一化
                ]
            )
        elif dataset_name in ["MNIST","FashionMNIST"]:
            transform = torchvision.transforms.ToTensor()
        
        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        # dataset not found exception
        error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]
    
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
    
    # split dataset according to iid flag
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )

        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]
    else:
        N_CLIENTS = num_clients
        DIRICHLET_ALPHA = alpha
        training_inputs = torch.Tensor(training_dataset.data)
        train_labels = np.array(training_dataset.targets)
        num_cls = len(training_dataset.classes)
        client_idcs = dirichlet_split_noniid(train_labels, alpha=DIRICHLET_ALPHA, n_clients=N_CLIENTS)

        local_datasets = []
        for client_index in client_idcs:
            # client_labels_tensor=torch.Tensor(client_labels)
            # print(client_labels_tensor)
            client_index_list = client_index.tolist()
            temp_data = []
            temp_labels = []
            for i in client_index_list:
                temp_data.append(training_inputs[i].tolist())
                temp_labels.append(train_labels[i].tolist())
            temp_data = torch.tensor(temp_data)
            temp_labels = torch.Tensor(temp_labels).long()
            temp = CustomTensorDataset(
                (
                    temp_data,
                    temp_labels
                ),
                transform=transform
            )
            local_datasets.append(temp)

    return local_datasets, test_dataset

reducer = UMAP(n_neighbors=10,
               # default 15, The size of local_noniid_2nn_mnist_100_10_3p-iid-3-client neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=3,  # default 2, The dimension of the space to embed into.
               metric='chebyshev',
               # default 'euclidean correlation chebyshev', The metric to use to compute distances in high dimensional space.
               n_epochs=350000,
               # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
               learning_rate=1.0,  # default 1.0, The initial learning rate for the embedding optimization.
               init='spectral',
               # default 'spectral（谱聚类）', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
               min_dist=0.05,  # default 0.1, The effective minimum distance between embedded points.（算a,b用的）
               spread=1.0,
               # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
               low_memory=False,
               # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
               set_op_mix_ratio=1.0,
               # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
               local_connectivity=1,
               # default 1, The local_noniid_2nn_mnist_100_10_3p-iid-3-client connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local_noniid_2nn_mnist_100_10_3p-iid-3-client level.
               repulsion_strength=1.0,
               # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
               negative_sample_rate=6,
               # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
               transform_queue_size=8.0,
               # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
               a=None,
               # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               b=None,
               # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
               random_state=42,
               # default: None, If int, random_state is the seed used by the random number generator;
               metric_kwds=None,
               # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
               angular_rp_forest=False,
               # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
               target_n_neighbors=-1,
               # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
               # target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different.
               # target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
               # target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
               transform_seed=42,
               # default 42, Random seed used for the stochastic aspects of the transform operation.
               verbose=False,  # default False, Controls verbosity of logging.
               unique=False,
               # default False, Controls if the rows of your data should be uniqued before being embedded.
               )
def umap_kmeans(matrix, sampled_client_indices):
    benign_result = []
    umap_result = reducer.fit_transform(matrix)
    y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(umap_result)
    v0 = []
    v1 = []
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            v0.append(sampled_client_indices[i])
        else:
            v1.append(sampled_client_indices[i])
    print(y_pred)
    y_pred_list=y_pred.tolist()
    print(sampled_client_indices)
    if len(v0) < len(v1):
        benign_result = v1
    elif len(v0) > len(v1):
        benign_result = v0
    else:
        benign_result =[4,5,6,7,8,9,10]
    print(benign_result)


    return benign_result,y_pred_list

def bron_kerbosch(r, p, x, max_cliques,graphs):
    if len(p) == 0 and len(x) == 0:
        max_cliques.append(r)
        return

    for v in list(p):  # 使用list()将集合转换为列表
        neighbors = set(graphs.neighbors(v))
        bron_kerbosch(r + [v], p.intersection(neighbors), x.intersection(neighbors), max_cliques,graphs)
        p.remove(v)
        x.add(v)

#
# writebook = xlwt.Workbook()  # 打开一个excel
# sheet1 = writebook.add_sheet('TCR_TPR')  # 在打开的excel中添加一个sheet
#
#
# def airm_geodesic_distance(A, B):
#     # Cholesky分解，A = LL^T，其中L是下三角矩阵
#     L = cholesky(A, upper=False)
#
#     # 计算矩阵L的逆
#     L_inv = torch.inverse(L)
#
#     # 转换为NumPy数组以使用logm函数
#     A_np = A.numpy()
#     B_np = B.numpy()
#
#     # 计算AIRM测地线距离
#     distance = torch.norm(torch.as_tensor(logm(np.dot(np.dot(L_inv, B_np), L_inv.T)), dtype=torch.float32), 'fro')
#     return distance


#计算cov矩阵
def torch_cov(input_vec:torch.tensor):
    input_vec = input_vec.reshape(1, len(input_vec))
    mean_value=torch.mean(input_vec,axis=1)
    x = input_vec-mean_value
    x_t=x.reshape(-1,1)
    cov_matrix = x_t @ x
    # cov_matrix = torch.mm(cov_matrix, cov_matrix.t())  # 使其对称
    # trace_cov = torch.trace(cov_matrix)
    # cov_matrix=cov_matrix/trace_cov
    n = cov_matrix.size(0)
    unit_matrix = torch.eye(n)
    result = cov_matrix + unit_matrix*0.5
    return result

def Log_matrix(P):
    # 对矩阵P进行特征值分解
    eigenvalues, eigenvectors = torch.linalg.eig(P)


    # 提取特征值和特征向量的实部（虚部通常很小可以忽略）
    real_eigenvalues = eigenvalues.real
    real_eigenvectors = eigenvectors.real

    # 特征值按从大到小排序
    sorted_eigenvalues, indices = torch.sort(real_eigenvalues, descending=True)
    sorted_eigenvectors = real_eigenvectors[:, indices]

    # 构造对角矩阵
    diag_matrix = torch.diag(sorted_eigenvalues)
    log_diag_matrix=torch.diag(torch.log(sorted_eigenvalues))

    # 正交化特征向量
    U, _ = torch.linalg.qr(sorted_eigenvectors, mode='reduced')

    # 检查正交性
    I = torch.eye(U.size(0))
    tolerance = 1e-6  # 设置一个容忍度
    orthogonal_test = torch.all(torch.abs(U @ U.t() - I) < tolerance)  # 使用容忍度来判断

    # if orthogonal_test:
    #     print("Orthogonal test passed.")
    # else:
    #     print("Orthogonal test failed.")

    # 检查对角化结果
    reconstructed_P = U @ diag_matrix @ U.t()
    log_p= U @ log_diag_matrix @ U.t()
    diagonalization_test = torch.allclose(reconstructed_P, P)  # 应该返回True
    # print(diagonalization_test)
    # print(reconstructed_P)
    # print(log_p)
    return log_p


def Exp_matrix(Q):
    # 对矩阵P进行特征值分解
    eigenvalues, eigenvectors = torch.linalg.eig(Q)


    # 提取特征值和特征向量的实部（虚部通常很小可以忽略）
    real_eigenvalues = eigenvalues.real
    real_eigenvectors = eigenvectors.real

    # 特征值按从大到小排序
    sorted_eigenvalues, indices = torch.sort(real_eigenvalues, descending=True)
    sorted_eigenvectors = real_eigenvectors[:, indices]

    # 构造对角矩阵
    diag_matrix = torch.diag(sorted_eigenvalues)
    log_diag_matrix=torch.diag(torch.exp(sorted_eigenvalues))

    # 正交化特征向量
    U, _ = torch.linalg.qr(sorted_eigenvectors, mode='reduced')

    # 检查正交性
    I = torch.eye(U.size(0))
    tolerance = 1e-6  # 设置一个容忍度
    orthogonal_test = torch.all(torch.abs(U @ U.t() - I) < tolerance)  # 使用容忍度来判断

    # if orthogonal_test:
    #     print("Orthogonal test passed.")
    # else:
    #     print("Orthogonal test failed.")

    # 检查对角化结果
    reconstructed_P = U @ diag_matrix @ U.t()
    log_p= U @ log_diag_matrix @ U.t()
    # diagonalization_test = torch.allclose(reconstructed_P, P)  # 应该返回True
    # print(diagonalization_test)
    # print(reconstructed_P)
    # print(log_p)
    return log_p


def distance_cal(A,B):#传入的是两个SPD流形，返回距离
    A=Log_matrix(A)
    B=Log_matrix(B)
    temp=A-B
    distance_A_B = torch.norm(temp, 'fro')
    return distance_A_B

def Lem_mean(m_matrix):#传入的是SPD流形,返回mean_valueSPD流形
    mean_value = np.zeros((3, 3))
    mean_value = torch.tensor(mean_value)
    for items in m_matrix:
        items = Log_matrix(items)
        mean_value = mean_value + items
        mean_value = mean_value / len(m_matrix)
    mean_value=Exp_matrix(mean_value)
    return mean_value

#计算每一个点到Lemmean的距离，返回距离向量金额距离向量的中值
def Median_distance(m_matrix):#传入的是良性的SPD空间，返回每一个SPD到均值的距离
    distances=[]
    #计算均值
    mean_value = Lem_mean(m_matrix)
    for items in m_matrix:
        distance=distance_cal(items,mean_value)
        distances.append(distance)
    distances=torch.tensor(distances)
    median_distances= torch.median(distances)

    return distances,median_distances


def Malicious_detector(matrix,sampled_client_indices,r):
    benign_result = []
    m_matrix = []
    umap_result = reducer.fit_transform(matrix)
    umap_result = torch.tensor(umap_result)
    #转化为SPD流形
    for items in umap_result:
        D_items = torch_cov(items)
        # A = torch.mm(A, A.t())  # 使其对称
        m_matrix.append(D_items)
    distances_median=[]

    distances,median_distance=Median_distance(m_matrix)
    for items in distances:
        temp=torch.abs(items-median_distance)
        distances_median.append(temp)
    distances_median=torch.tensor(distances_median)
    for items in distances_median:
        pass
    # if r in [1,10,15,20]:
    #     malicious_values, malicious_index = torch.topk(distances_median, 9)
    # else:
    malicious_values, malicious_index = torch.topk(distances_median, 6)

    malicious_index=malicious_index.tolist()
    benign_index=[]
    for indexes in sampled_client_indices:
        if indexes in malicious_index:
            pass
        else:
            benign_index.append(indexes)
    print("distance median")
    print(distances_median)




    return malicious_index,benign_index,distances_median

def manifold_sim(distance_median):
    temp=1+torch.log(1+distance_median)
    temp=1/temp
    return temp

def Malicious_detectorv1(matrix,sampled_client_indices,r):
    benign_result = []
    malicious_result=[]
    m_matrix = []
    umap_result = reducer.fit_transform(matrix)
    umap_result = torch.tensor(umap_result)
    #转化为SPD流形
    for items in umap_result:
        D_items = torch_cov(items)
        # A = torch.mm(A, A.t())  # 使其对称
        m_matrix.append(D_items)
    #计算每一个用户和其他用户之间的距离
    distance_mult=[]
    for i in range(len(m_matrix)):
        temp=[]
        for j in range(len(m_matrix)):
            if i!=j:
                distance_temp = distance_cal(m_matrix[i], m_matrix[j])
                temp.append(distance_temp.item())
        temp=torch.tensor(temp)
        distance_mult.append(temp)

    # print("distance_mult")
    # print(distance_mult)
    acc_distance=[]
    for items in distance_mult:
        item=items.reshape(-1,1)
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(item)
        temp_acc = 0
        for i in range(len(y_pred)):

            if y_pred[i]==1:
                temp_acc=temp_acc+items[i]
            else:
                pass
        acc_distance.append(temp_acc)
    print("acc distance")
    print(acc_distance)
    acc_distance=torch.tensor(acc_distance)
    acc_distance=acc_distance.reshape(-1,1)
    result_pred = KMeans(n_clusters=2, random_state=9).fit_predict(acc_distance)
    v0 = []
    v1 = []
    for i in range(len(result_pred)):
        if result_pred[i] == 0:
            v0.append(sampled_client_indices[i])
        else:
            v1.append(sampled_client_indices[i])
    if len(v0) < len(v1):
        benign_result = v1
        malicious_result=v0
    elif len(v0) > len(v1):
        benign_result = v0
        malicious_result = v1
    else:
        benign_result = sampled_client_indices
    print(benign_result)









    return m_matrix,benign_result,malicious_result,result_pred




def Reputation(m_matrix,benign_result,sampled_client_indices):
    reputation=[]
    benign_matrix=[]
    sim=[]
    for items in sampled_client_indices:
        if items in benign_result:
            benign_matrix.append(m_matrix[items])
    distances,median_distance=Median_distance(benign_matrix)
    for simitem in distances:
        temp_sim=manifold_sim(simitem)
        sim.append(temp_sim.item())
    sim=torch.tensor(sim)
    weights=torch.nn.functional.softmax(sim,dim=0)
    weights=weights.tolist()


    return weights



def current_reputation(m_matrix,benign_result,sampled_client_indices):
    reputation = []
    benign_matrix = []
    distances=[]
    sim = []
    for items in sampled_client_indices:
        if items in benign_result:
            benign_matrix.append(m_matrix[items])
    lem_mean=Lem_mean(benign_matrix)#良性用户的黎曼均值

    for items in m_matrix:
        # items=Log_matrix(items)
        distance=distance_cal(items,lem_mean)
        distances.append(distance)
    for simitem in distances:
        temp_sim=manifold_sim(simitem)
        sim.append(temp_sim.item())
    sim=torch.tensor(sim)
    softmax_repu=torch.nn.functional.softmax(sim,dim=0)

    return softmax_repu





