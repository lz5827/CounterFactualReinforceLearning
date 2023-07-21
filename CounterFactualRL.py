# import Packages
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import torch
from torch import nn, optim
import torchvision
import pandas as pd
import time
import argparse
np.set_printoptions(precision = 4) # 设置小数精度
np.set_printoptions(suppress = True)
import copy
import pickle
from rich_fmt import rich_fmt
from rich.console import Console
from detection import AD_CNN_Limit
from rich.table import Column, Table
from data_utils import load_GPU_data
console = Console()
import torch.multiprocessing as mp  # multi-threads
from model.cnn import CNN
from model.ae import AE
device = "cuda"



 # 计算传过来的参数的损失值
def get_loss_func(Xt, model, loss_type):
    model.train()
    model.eval()
    loss = np.zeros((Xt.size(0),1))
    with torch.no_grad():
        rec, z, mu, logvar = model(Xt)
        loss_vae_res = ((Xt - rec) **2).mean(-1).mean(-1).mean(-1)
        KLD = 0.5*(mu.pow(2) + logvar.exp() - logvar - 1).mean(-1) # KL损失
        
        _loss_vae_res = loss_vae_res.cpu().detach().numpy()
        _KLD = KLD.cpu().detach().numpy()
    if loss_type == "loss_res":
        loss = _loss_vae_res
    else:
        loss = _KLD
    return loss

def generate_z(AE_Encoder, x):
    for param in AE_Encoder.parameters():
        param.requires_grad = False
    z = AE_Encoder(x)
    return z

def generate_x_cf(AE_Decoder,z):
    for param in AE_Decoder.parameters():
        param.requires_grad = False
    x_cf = AE_Decoder(z)  
    return x_cf



def actor(process_name, model, AE_Encoder, AE_Decoder, lra, epoches, loss_type, q_feed, q_back):
    while True:
        task = q_feed.get()
        if isinstance(task, int) and task < 0:
            break
            
        reward = 0
        i_window, xt_window, z_init_delta, loss_origin, loss_goal = task
        window = torch.from_numpy(xt_window).to(device)
        z_delta = torch.from_numpy(z_init_delta).to(device)
        # print("z_delta形状", np.shape(z_delta))
        # 故障数据的 xt--> AE_Encoder--> z
        # print(window.shape, type(window))
        z = generate_z(AE_Encoder, window) #改为 RBC形式
        z_delta.requires_grad = True
        opt_z_delta = optim.Adam([z_delta], lr = lra)
        
        for epoch in range(epoches):
            # time.sleep(5)
            # print("z_delta", z_delta,'\n', "z_delta均值", torch.mean(z_delta))
            # print("z", z, '\n', "z均值", torch.mean(z))
            # print("epoches", epoch)
            z_cf = z_delta + z
            # print("z_cf",z_cf,'\n',"z", z, '\n', 'z_delta', z_delta)
            # print("z_cf", torch.mean(z_cf))
            x_cf = generate_x_cf(AE_Decoder, z_cf) # 多了这个
            rec, zz, mu, logvar  = model(x_cf)
            if loss_type == "loss_res":
                loss = ((x_cf - rec) **2).mean(-1).mean(-1).mean(-1)
            else:
                loss = 0.5*(mu.pow(2) + logvar.exp() - logvar - 1).mean(-1)
                # print("loss_kld")
            # print('loss值',loss)
            loss = loss.mean()
            loss.backward()
            opt_z_delta.step()
            optim_loss = loss.detach().cpu().numpy()
            # if epoch == 100:  # 输出查看
            #     time.sleep(5)
            #     print("z", z, '\n', "z均值", torch.mean(z))
            #     print("z_delta", z_delta,'\n', "z_delta均值", torch.mean(z_delta))
            
            # reward = critic(optim_loss, loss_goal, loss_origin, init_reward = reward)
            # print("Optim_loss", optim_loss, "loss_goal", loss_goal)
            if optim_loss < loss_goal:
                break
            
        if optim_loss < loss_goal:  # 小于指定的目标   
            state = 'Done'
        else:
            if optim_loss < loss_origin:
                state = 'Better'
            else:
                state = 'Worse'
        z_cf = z_delta.detach().cpu().numpy() # 不改变网络的参数，将delta保存到cpu     改为RBC
        x_cf = x_cf.detach().cpu().numpy()
    
        q_back.put([state, i_window, epoch, z_cf, x_cf, optim_loss, loss_goal, reward])
    
  
    # z_delta = 0
    # zcf = z + z_delta
    # x_cf = self.Decoder(zcf)


def critic(optim_loss, loss_goal, loss_origin, init_reward = 0):
    
    reward = init_reward  # 根据上一个相加
    # 前一时刻的reward
    # 这个loss 与 当前的时刻进行比较得出分数评价
    # 要求算一个初始的 loss_delta
    # self.reward_coff
    reward = init_reward + 0.1*(loss_origin-optim_loss)/np.abs(loss_goal-loss_origin)        
    
    return reward
   
if __name__ == '__main__':
    #  主函数   定义诊断开始 
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='CNNVAE Loss optimization based diagnosis')
    parser.add_argument('-t', "--threads", type=int, default=2) 
    parser.add_argument('-f', "--fault", type=int, default="0") # 默认故障0<其实是1>
    parser.add_argument('-l', "--loss_type", type=str, default="loss_res", choices=['loss_res', 'loss_kld'])
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3) 
    parser.add_argument('-me',"--max_epoches", type=int, default = 1000)
    args = parser.parse_args()
    console.print('[b][red]Diagnosis begin!')
    
    threads = args.threads
    fault_num = args.fault
    max_epoches = args.max_epoches
    loss_type = args.loss_type
    lr = args.learning_rate
    
    # 加载模型等参数
    """
    以下参数跟训练集加载数据相同，均为加载数据超参数
    """
    num_train=100
    num_test=12
    slide_size= 4
    # AE 浅空间神经元个数
    Latent_Dim = 30
    FAR = 0.01
    # 加载需要诊断数据
    X, Xt = load_GPU_data(num_train, num_test,slide_size,fault_num)
    ## Load Predictor 
    CNN_model = torch.load('experiment/trained_model/model_0075.pth')
    ad = AD_CNN_Limit(CNN_model, X, Xt, FAR)
    
    model = CNN_model.share_memory()
    model.to(device)
    ## model 参数固定
    for param in model.parameters():
        param.requires_grad = False
    """
    thres 阈值
    test_index 统计量
    按列  0：res；  1：kld
    """
    thres, test_index = ad.calc("")
    if loss_type == "loss_res":
        index = 0
    else: 
        index = 1
    Y_t = thres[:,index]  # 计算出目标控制限
    
    """
    还需定义
    空白函数 Z_delta
    X_cf 生成函数
    优化过程
    
    """
    os.makedirs("experiment/Reinforce_diagnosis/", exist_ok=True)
    fn_npz = 'experiment/Reinforce_diagnosis/%s_F%02d_%s.npz' % (loss_type, fault_num + 1, max_epoches)
    fn_Z_cf_npz = 'experiment/Reinforce_diagnosis/%s_Z_cf_F%02d_%s.npz' % (loss_type, fault_num + 1, max_epoches)
    if os.path.exists(fn_npz): # 判断是否诊断 诊断后直接return 0 
        console.print("[b][red]Skip! Because it already finished[red]", fn_npz)
        # X_cf = load_X_cf(fc_npz)
        # return 0# X_cf
    else:
        console.print("[b][blue]Will run and save to[/blue] [yellow]%s[/yellow]"%(fn_npz)) 
    
        if max_epoches>1000:
            load_Z_cf_npz = 'experiment/Reinforce_diagnosis/%s_Z_cf_F%02d_%s.npz' % (loss_type, fault_num + 1, max_epoches-1000)
            
            if os.path.exists(load_Z_cf_npz):
                console.print("[b]Starting from [green]%d[/green], load Z_cf init from [yellow]%s[/yellow]"%(max_epoches - 1000, load_Z_cf_npz))
                Z_init_delta = np.load(load_Z_cf_npz)['Z_cf']
                epoches_to_do = 1000
            else:
                console.print("[b]We don't have[res]%s[/res],you should finisih  previous results"%(load_Z_cf_npz))
        else:
            console.print('[b]Starting from scrach, will save to [blue]%s[/blue]'%(fn_Z_cf_npz))
            Z_init_delta = np.zeros((Xt.size(0), Latent_Dim), dtype = np.float32)
            epoches_to_do = max_epoches
        
        ## 定义AE
        
        # model_path  = './experiment/AE_trained_model/'
        # os.makedirs(model_path, exist_ok=True)
        # ae_path = model_path
        # ae = AE()
        # ae.to(device)
        # param_list=[]
        # best_wts_list=[]
        # for tmp_sub_model in ae.AE_list:  # 添加到最好的模型集合
        #     best_wts_list.append(tmp_sub_model.state_dict())
        #     for tmp_param in tmp_sub_model.parameters():    # 参数
        #         param_list.append(tmp_param)
        # if len(os.listdir(ae_path)) == 0 : # 是否训练？
        #     Cond_Loss = torch.nn.CrossEntropyLoss(reduction='none')
        #     Opt_ae = optim.Adam(param_list, lr=0.001)
        #     for epoch in range(1, 500+1):
        #         #  开始训练
        #         ae.train()
        #         for i_mb in range(0, X.shape[0], 200): 
        #             x = X[i_mb:i_mb + 200]
        #             Opt_ae.zero_grad()
        #             hat_x = ae(x) 
        #             loss = ((hat_x - x) **2).mean(-1).mean(-1).mean(-1)   # 求解重构均方误差
        #             loss = loss.mean()
        #             loss.backward()
        #             Opt_ae.step() 
        #         if epoch % 10 == 0 and epoch !=0:
        #             console.print(str(epoch))
        #             console.print('loss=%.6f'%(loss.item()), end='\r')
        #             torch.save(ae, './experiment/AE_trained_model/AE_model_%04d.pth'%(epoch))
        # else:
        #     ae = torch.load('experiment/AE_trained_model/AE_model_0150.pth')
        
        
        # AE 
        ae = torch.load('experiment/AE_trained_model/AE_model_0150.pth')
        ae.to(device)
        ae_Encoder = ae.AE_list[0].share_memory()
        ae_Decoder = ae.AE_list[1].share_memory()
        
        
        # 计算优化过程  -->多线程
        # 出现传参数出现问题
        loss_origins = get_loss_func(Xt, model, loss_type)
        processes, q_feed, q_back = [], mp.Queue(), mp.Queue()
        for p_id in range(threads):
            p = mp.Process(target = actor, args = ('P_%d'%(p_id), model, ae_Encoder, ae_Decoder, lr, epoches_to_do, loss_type, q_feed, q_back))
            p.start()
            processes.append(p)
            
        # 创建多多任务
        num_tasks = Xt.size(0)
        for i_window in range(num_tasks):
            xt_window = Xt[i_window:i_window+1].cpu().numpy().copy()
            z_init_delta = Z_init_delta[i_window:i_window+1].copy()
            loss_origin = loss_origins[i_window].copy()  # 可能不是一个矩阵，而是一维向量
            loss_goal = Y_t[i_window].copy()
            
            q_feed.put([i_window, xt_window, z_init_delta, loss_origin, loss_goal])
        # print("是否完成创建多任务") 
        Z_cf = np.zeros((Xt.size(0), Latent_Dim), dtype = np.float32) 
        X_cf = np.zeros(Xt.shape, dtype = np.float32)
        c = {'Done':'green','Better':'yellow','Worse':'red'}
        overall_state = {'Done':0, 'Better':0, 'Worse':0}
        for i in range(num_tasks):
            ## 出现在这错误
            state, i_window, epoch, z_cf, x_cf, optim_loss, loss_goal, reward = q_back.get()
            overall_state[state] +=1
            
            if state!='Worse':
                X_cf[i_window:i_window+1] = x_cf
                Z_cf[i_window:i_window+1] = z_cf
                
            else: 
                print('skip')
            t1 = '> %s, i:%3d, [%s]%7s[/%s] '%(loss_type, i_window, c[state], state, c[state])
            t2 = ' epoch:%4d, loss_orig: %.2f -->> optim_loss %.2f, goal: %.2f'%(epoch, loss_origins[i_window], optim_loss, loss_goal)
            
            console.print(t1+t2, end = '\r')
        print(overall_state)
        
        for i_window in range(threads):
            q_feed.put(-1)
        for process in processes:
            process.join()
        np.savez(fn_npz, X_cf = X_cf)
        np.savez(fn_Z_cf_npz, Z_cf = Z_cf)
        # X_cf = load_X_cf(fc_npz)
        # return X_cf