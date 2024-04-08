import copy
import gc
import logging

import numpy as np
import torch
import torch.nn as nn
import xlrd
import xlutils as xlutils
import xlwt
import datetime
from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
# import tqdm
from collections import OrderedDict
import xlrd
import xlutils as xlutils
import xlwt
import datetime
from .models import *
from .utils import *
from .client import Client
import random
logger = logging.getLogger(__name__)
import math
writebook = xlwt.Workbook()  # 打开一个excel
sheet_epoch = writebook.add_sheet('select')
writebook2 = xlwt.Workbook()  # 打开一个excel
sheet_epoch2 = writebook2.add_sheet('tcr_tpr')# 在打开的excel中添加一个sheet



class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients(选择一部分客户),
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients（新选择的客户） will recevie the updated global model as its local model.
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.（tensorboard用来记录）
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).#True or False
        init_config: kwargs(参数) for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.（每一轮中选择客户的比例）
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.（客户在本地训练的次数）
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: kwargs(参数) provided for optimizer.
    """
    def __init__(self, writer, model_config={}, global_config={}, data_config={}, init_config={}, fed_config={}, optim_config={}):
        self.clients = None
        self._round = 0
        self.writer = writer

        # self.model = ResNet18()
        # if not hasattr(self.model, 'name'):
        #     setattr(self.model, 'name', 'resnet ')
        self.model = eval(model_config["name"])(**model_config)

        self.M = []#记录每一轮中的每一个用户累积的不公平度
        self.repu=[]#记录每一轮中用户的历史名誉
        self.choice_times=[]#记录客户被选中的次数


        # self.user_reputation={}
        self.communication_round=0
        self.random_list = []

        #全局配置
        self.seed = global_config["seed"]
        self.device = global_config["device"]
        self.mp_flag = global_config["is_mp"]

        #数据配置
        self.data_path = data_config["data_path"]
        self.dataset_name = data_config["dataset_name"]
        self.num_shards = data_config["num_shards"]
        self.iid = data_config["iid"]
        self.alpha=data_config["alpha"]

        self.init_config = init_config

        #联邦学习中本地参数设置
        self.fraction = fed_config["C"]
        self.num_clients = fed_config["K"]
        self.num_rounds = fed_config["R"]
        self.local_epochs = fed_config["E"]
        self.batch_size = fed_config["B"]

        #损失函数和优化器设置
        self.criterion = fed_config["criterion"]
        self.optimizer = fed_config["optimizer"]
        self.optim_config = optim_config
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)#设置CPU生成随机数的种子

        init_net(self.model, **self.init_config)#初始化网络

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid, self.alpha)
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size,
            criterion=self.criterion, num_local_epochs=self.local_epochs,
            optimizer=self.optimizer, optim_config=self.optim_config
            )
        
        # send the model skeleton to all clients
        self.transmit_model()#第一次发送，所以直接发给所有的用户
        
    def create_clients(self, local_datasets):#创建client,分配数据
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):#对create_clients中创建的clients进行配置
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:#第一次训练之前给所有客户发送模型
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for client in tqdm(self.clients, leave=False):#对于clients[]中每一个用户，将模型model复制给它
                client.model = copy.deepcopy(self.model)

            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:#在每一轮训练过程中给选中的客户发送模型
            # send the global model to selected clients
            assert self._round != 0

            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)#self.fraction * self.num_clients是选择的客户数量
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())
        sampled_client_indices=[i for i in range(0,30)]
        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices):
        """Call "client_update" function of each selected client."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update()
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()


        return selected_total_size#selected_total_size是客户拥有样本的数量
    
    def mp_update_selected_clients(self, selected_index):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[selected_index].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        r=self.communication_round
        macilious_list=self.random_list
        self.clients[selected_index].client_update(r,macilious_list)
        client_size = len(self.clients[selected_index])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[selected_index].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size



    def average_model(self, sampled_client_indices, coefficients,r):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()


        averaged_weights = OrderedDict()

        #LHS 8.23
        #将本地模型展成一维tensor
        matrix = []
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_model = self.clients[idx].model
            matrix.append(torch.nn.utils.parameters_to_vector(local_model.parameters()).tolist())
        matrix = np.array(matrix)


        # malicious_index,benign_index,distance_median=Malicious_detectorv1(matrix,sampled_client_indices,r)
        m_matrix,benign_index,malicious_index,result_pred= Malicious_detectorv1(matrix, sampled_client_indices, r)
        result_pred=result_pred.tolist()
        for i in range(len(result_pred)):
            sheet_epoch2.write(r, i, result_pred[i])  # 写入excel，i行0列
        writebook2.save('tcr_tpr.xls')  # 一定要记得保存
        print("本轮良性用户")
        print(benign_index)
        # 用户本轮评分
        sim=current_reputation(m_matrix,benign_index,sampled_client_indices)
        print("sim")
        print(sim)


        # #基于历史行为的良性用户模型评价
        reputation_list=[]
        if r ==0:
            for indexes in sampled_client_indices:
                client_reputation = {}
                client_reputation[indexes] = [0,0]
                if indexes in malicious_index:
                    client_reputation[indexes][0] = 1
                else:
                    client_reputation[indexes][1]=1
                reputation_list.append(client_reputation)
            self.repu.append(reputation_list)

        else:
            for indexes in sampled_client_indices:
                client_reputation = self.repu[r-1][indexes]
                if indexes in malicious_index:
                    client_reputation[indexes][0] = client_reputation[indexes][0]+1
                else:
                    client_reputation[indexes][1]=client_reputation[indexes][1]+1
                reputation_list.append(client_reputation)
            self.repu.append(reputation_list)

        # 用户历史行为评分
        output=[]
        current_state=self.repu[r]
        for items in current_state:
            temp=items.values()
            temp = list(temp)[0]

            reputation=(temp[1]+1)/(temp[0]+temp[1]+2)
            output.append(reputation)
        output=torch.tensor(output)

        # #用户本轮评分
        # sim=[]
        # for items in distance_median:
        #     temp=manifold_sim(items)
        #     sim.append(temp)
        # print("sim")
        # print(sim)












        # #基于历史行为的良性用户模型评价
        # matrix=torch.tensor(matrix)
        # global_model_vector=torch.mean(matrix,dim=0)
        # # # 评估本地模型的质量并打分
        # output = torch.cosine_similarity(matrix, global_model_vector)


        #基于Lyapunov的客户选择
        CSI = []
        x = []
        selected_client_indices=[]
        #第0轮M队列初始化为全0
        if r == 0:
            temp = [0 for i in range(len(sampled_client_indices))]
            self.M.append(temp)


        #计算CSI值以及选择CSI最大的n个用户
        control_parameter=1.0
        for i in range(len(sampled_client_indices)):
            CSI.append(control_parameter*output[i]*sim[i]+self.M[r][i])
        #不选择恶意用户
        for i in range(len(CSI)):
            if i in malicious_index:
                CSI[i]=-100
        CSI = np.array(CSI)


        #选择客户端的数量
        n = 15
        top_indices = np.argsort(CSI)[-n:][::-1]

        #计算客户端选择向量x_i
        for i in range(len(sampled_client_indices)):
            if i in top_indices:
                x.append(1)
                selected_client_indices.append(sampled_client_indices[i])
            else:
                x.append(0)


        #计算下一轮M队列

        discount_factor=1.0
        temp=[]
        for i in range(len(sampled_client_indices)):
            # Q = self.M[r][i] + (1 - x[i]) * output[i] * sim[i] * discount_factor - x[i]
            if i in malicious_index:
                Q = self.M[r][i] -1
            else:
                Q = self.M[r][i] + (1 - x[i]) * output[i]*sim[i] * discount_factor - x[i]



            Q=max(0,Q)
            temp.append(Q)
        self.M.append(temp)

        #聚合得到全局模型
        selected_client=[]
        for items in sampled_client_indices:
            if items in selected_client_indices:
                selected_client.append(1)
            else:
                selected_client.append(0)
        # sheet_epoch.write(r, 2, r + 1)  # 写入excel，i行0列
        for i in range(len(selected_client)):
            sheet_epoch.write(r, i, selected_client[i])  # 写入excel，i行0列、
        writebook.save('select.xls')  # 一定要记得保存

        print("selected_client_indices")
        print(selected_client_indices)

        for it, idx in enumerate(selected_client_indices):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        print("本轮队列M：")
        print(self.M[r])
        # matrix = []
        # for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
        #     local_weights = torch.nn.utils.parameters_to_vector(self.clients[idx].model.parameters()).tolist()
        #     matrix.append(local_weights)
        # matrix = np.array(matrix)
        # benign_sample, y_pred = umap_kmeans(matrix, sampled_client_indices)
        # for i in range(len(y_pred)):
        #     sheet_epoch.write(r, i, y_pred[i])  # 写入excel，i行0列
        #     writebook.save('TCR_TPR.xls')  # 一定要记得保存
        #
        # averaged_weights = OrderedDict()
        # for it, idx in tqdm(enumerate(benign_sample), leave=False):
        #     local_weights = self.clients[idx].model.state_dict()
        #     for key in self.model.state_dict().keys():
        #         if it == 0:
        #             averaged_weights[key] = coefficients[it] * local_weights[key]
        #         else:
        #             averaged_weights[key] += coefficients[it] * local_weights[key]

        self.model.load_state_dict(averaged_weights)








        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()

        for idx in sampled_client_indices:
            self.clients[idx].client_evaluate()

        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def mp_evaluate_selected_models(self, selected_index):
        """Multiprocessing-applied version of "evaluate_selected_models" method."""
        self.clients[selected_index].client_evaluate()
        return True

    def train_federated_model(self,r):
        """Do federated training."""
        # select pre-defined fraction of clients randomly 挑选用户
        sampled_client_indices = self.sample_clients()

        # send global model to the selected clients 发送模型
        self.transmit_model(sampled_client_indices)

        # updated selected clients with local dataset
        if self.mp_flag:
            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
            selected_total_size = sum(selected_total_size)
        else:
            selected_total_size = self.update_selected_clients(sampled_client_indices)

        # evaluate selected clients with local dataset (same as the one used for local update)
        if self.mp_flag:
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
            print(message); logging.info(message)
            del message; gc.collect()

            with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
                workhorse.map(self.mp_evaluate_selected_models, sampled_client_indices)
        else:
            self.evaluate_selected_models(sampled_client_indices)






        # calculate averaging coefficient of weights
        # print("************************************")
        # print(selected_total_size) #6000所有被选择用户（10个）数据量的加和
        # print([len(self.clients[idx]) for idx in sampled_client_indices])
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]
        # print(mixing_coefficients)#[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients,r)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)

        test_loss, correct = 0, 0
        with torch.no_grad():#对整个联邦学习模型进行测试
            for data, labels in self.dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)
                outputs = self.model(data)
                test_loss += eval(self.criterion)()(outputs, labels).item()
                
                predicted = outputs.argmax(dim=1, keepdim=True)
                correct += predicted.eq(labels.view_as(predicted)).sum().item()#相加因为后面要计算平均值
                
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to("cpu")

        test_loss = test_loss / len(self.dataloader)
        test_accuracy = correct / len(self.data)
        return test_loss, test_accuracy

    def fit(self):
        """Execute the whole process of the federated learning."""
        self.results = {"loss": [], "accuracy": []}

        n = 3  # 每一轮中攻击者的数量
        min_value = 1
        max_value = 10
        writebook3 = xlwt.Workbook()  # 打开一个excel
        sheet_epoch3 = writebook3.add_sheet('ACC')  # 在打开的excel中添加一个sheet
        for r in range(self.num_rounds):#控制联邦学习迭代的轮数
            self._round = r + 1
            self.communication_round=r

            # self.random_list = []#每一轮中攻击者的index
            # while len(self.random_list) < n:
            #     random_num = random.randint(min_value, max_value)
            #     if random_num not in self.random_list:
            #         self.random_list.append(random_num)

            # print("macilious", self.random_list)


            
            self.train_federated_model(r)
            test_loss, test_accuracy = self.evaluate_global_model()
            sheet_epoch3.write(r, 1, r)
            sheet_epoch3.write(r, 2, test_accuracy)


            writebook3.save('ACC.xls')  # 一定要记得保存
            
            self.results['loss'].append(test_loss)
            self.results['accuracy'].append(test_accuracy)

            self.writer.add_scalars(
                'Loss',
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Accuracy', 
                {f"[{self.dataset_name}]_{self.model.name} C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
                self._round
                )

            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()
