# Program description: Calculate the outflow interval of Shuibuya and Diheyan
# Only Shuibuya and Diheyan
# Input: time series 1, inbound flow (Shuibuya and water - diaphragm interval inflow) 2, previous upstream water level 2, previous outbound 2
# Output: outgoing flow upper and lower bounds, using the loss function to establish a link with the actual value
# Model: LSTM
# Application: simulate historical outflow from reservoir, all X inputs, get outputs, simultaneous segments

# -*- coding: utf-8 -*-
"""
@author: yalianzheng
"""


import numpy as np   
import pandas as pd 
import torch
import torch.utils.data as Data
import torch.nn as nn
import time
import random
import math
import copy

since = time.time()   

EPOCHES = 300        
INPUT_SIZE = 19        
TIME_STEP = 20         
NUM_LAYERS = 2        
HIDDEN_SIZE = 32       
OUT_SIZE = 4           
BATCH_SIZE = 300       
LR = 0.001             
his_days = TIME_STEP-1            
repeatrun = 6
bei = 5
PINC = 0.8
w1 = 1  
w2 = 1
w3 = 1
w4 = 1

filePath_daily = r'./data_input.xlsx'
out_path = r'./data_generator'

day_number0 = np.array(pd.read_excel(filePath_daily, header=None))[:,1]  

sby_zup0 = np.array(pd.read_excel(filePath_daily, header=None))[:,2]  
ghy_zup0 = np.array(pd.read_excel(filePath_daily, header=None))[:,3]  

sby_zdown0 = np.array(pd.read_excel(filePath_daily, header=None))[:,4]
ghy_zdown0 = np.array(pd.read_excel(filePath_daily, header=None))[:,5] 

sby_inflow_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,6]  
sg_inflow_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,7] 
gg_inflow_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,8]  

sby_inflow_predic1_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,9] 
sg_inflow_predic1_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,10]  
gg_inflow_predic1_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,11] 

sby_e_outflow_pre0 = np.array(pd.read_excel(filePath_daily, header=None))[:,12] 
sby_outflow_pre0 = np.array(pd.read_excel(filePath_daily, header=None))[:,13] 
ghy_e_outflow_pre0 = np.array(pd.read_excel(filePath_daily, header=None))[:,14] 
ghy_outflow_pre0 = np.array(pd.read_excel(filePath_daily, header=None))[:,15]  

n_qj_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,16]  
n_tgr_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,17]  
temp_min_data0 = np.array(pd.read_excel(filePath_daily, header=None))[:,18]  
temp_max_data0  = np.array(pd.read_excel(filePath_daily, header=None))[:,19] 

sby_dis0 = np.array(pd.read_excel(filePath_daily, header=None))[:,20]
ghy_dis0 = np.array(pd.read_excel(filePath_daily, header=None))[:,21] 

sby_dis0_max = sby_dis0.max()  
sby_dis0_min = sby_dis0.min()

ghy_dis0_max = ghy_dis0.max()  
ghy_dis0_min = ghy_dis0.min()

def mmn(aa):
    a_max = aa.max()
    a_min = aa.min()
    aa = (aa - a_min) / (a_max - a_min)
    return aa

def PICP(a_max, a_min,  y_obs):
    C = []
    stepp = len(a_max)
    for kkk in range(stepp):
        if a_min[kkk, 0] < y_obs[kkk, 0] < a_max[kkk, 0]:
           C.append(1)
        else:
           C.append(0)
    average_C = sum(C) / len(C)
    return average_C

def PINRW(a_max, a_min):
    L = a_max - a_min
    L2 = L*L
    m = torch.mean(torch.from_numpy(L2))
    n = max(L)
    LL = m**0.5/n
    return LL

def PIAW(a_max, a_min): #PIAW
    L = a_max - a_min
    m = torch.mean(torch.from_numpy(L))
    return m

def PINAW(a_max, a_min):  #PINAW
    L = a_max - a_min
    m = torch.mean(torch.from_numpy(L))
    n = max(L)
    LL = m/n
    return LL

def PIMSE(a_max, a_min,  y_obs):
    XX = (a_max - y_obs)*(a_max - y_obs)+(a_min - y_obs)*(a_min - y_obs)
    X = torch.from_numpy(XX).float()
    RESULT = torch.mean(X)
    return RESULT

def PINMSE(a_max, a_min,  y_obs):
    pp = ((a_max - y_obs)*(a_max - y_obs)+(a_min - y_obs)*(a_min - y_obs))/(y_obs*y_obs)
    ppp = torch.from_numpy(pp).float()
    RESULTp = torch.mean(ppp)
    return RESULTp

def PICD(a_max, a_min,  y_obs):
    SS =((a_max+a_min)*0.5 - y_obs)**2
    S = torch.from_numpy(SS).float()
    RESULT2 = torch.mean(S)
    return RESULT2

def NCWC(PINAW,PICP):
    u = PINC
    beita = 6
    afa = 0.1
    ita = 15
    if PICP >= u:
        cwc_proposed = beita*PINAW
    else:
        cwc_proposed = (afa+beita*PINAW)*(1+math.exp(-ita*(PICP-u)))
    return cwc_proposed

day_number = mmn(day_number0)

sby_zup = mmn(sby_zup0)
ghy_zup = mmn(ghy_zup0)

sby_zdown = mmn(sby_zdown0)
ghy_zdown = mmn(ghy_zdown0)

sby_inflow_data = mmn(sby_inflow_data0)
sg_inflow_data = mmn(sg_inflow_data0)
gg_inflow_data = mmn(gg_inflow_data0)

sby_inflow_predic1_data = mmn(sby_inflow_predic1_data0)
sg_inflow_predic1_data = mmn(sg_inflow_predic1_data0)
gg_inflow_predic1_data = mmn(gg_inflow_predic1_data0)

sby_e_outflow_pre = mmn(sby_e_outflow_pre0)
sby_outflow_pre = mmn(sby_outflow_pre0)
ghy_e_outflow_pre = mmn(ghy_e_outflow_pre0)
ghy_outflow_pre = mmn(ghy_outflow_pre0)

n_qj_data = mmn(n_qj_data0)
n_tgr_data = mmn(n_tgr_data0)
temp_min_data = mmn(temp_min_data0)
temp_max_data = mmn(temp_max_data0)

sby_dis = mmn(sby_dis0)
ghy_dis = mmn(ghy_dis0)

X = np.c_[day_number, sby_zup, ghy_zup, sby_zdown, ghy_zdown, sby_inflow_data, sg_inflow_data, gg_inflow_data, sby_inflow_predic1_data, sg_inflow_predic1_data, \
    gg_inflow_predic1_data, sby_e_outflow_pre, sby_outflow_pre, ghy_e_outflow_pre, ghy_outflow_pre, n_qj_data, n_tgr_data, temp_min_data, temp_max_data]
Y = np.c_[sby_dis, ghy_dis]

X = X.astype(np.float32)  
Y = Y.astype(np.float32)  
print(len(Y))

X = X[:(len(X)-lead_t),:]  
Y = Y[his_days:(len(Y)),:]  

for i in range(0, X.shape[0]-(his_days), 1):  
    if i == 0:
        operation = X[:TIME_STEP, :]
        operation_data = np.r_[operation]
    else:
        operation = X[i:i+TIME_STEP, :]
        operation_e = np.r_[operation]
        operation_data = np.vstack((operation_data, operation_e))  

X_ten = torch.from_numpy(operation_data.reshape(-1, TIME_STEP, INPUT_SIZE)).float()  
Y_ten = torch.from_numpy(Y.reshape(-1, 2)).float() 

real_len = Y_ten.shape[0]               
total_len = real_len-his_days          
valid_len = 731  
test_len = 1096  
train_len = total_len-test_len-valid_len        

X_train = X_ten[:train_len+his_days, :, :]
Y_train = Y_ten[:train_len+his_days, :]
X_valid = X_ten[train_len+his_days:train_len+his_days+valid_len, :, :]
Y_valid = Y_ten[train_len+his_days:train_len+his_days+valid_len, :]
X_test = X_ten[train_len+his_days+valid_len:, :, :]
Y_test = Y_ten[train_len+his_days+valid_len:, :]

opt_re_y_boundary_test = np.zeros(shape=[test_len, 6])
opt_re_y_boundary_valid = np.zeros(shape=[valid_len, 6])
opt_re_y_boundary_train = np.zeros(shape=[train_len + his_days, 6])

opt_score_sby = np.zeros(shape=[3, 8])  
opt_score_ghy = np.zeros(shape=[3, 8])  

all_score_sby_train = np.zeros(shape=[EPOCHES, 8])  
all_score_sby_valid = np.zeros(shape=[EPOCHES, 8])  
all_score_sby_test = np.zeros(shape=[EPOCHES, 8])

all_score_ghy_train = np.zeros(shape=[EPOCHES, 8]) 
all_score_ghy_valid = np.zeros(shape=[EPOCHES, 8])  
all_score_ghy_test = np.zeros(shape=[EPOCHES, 8])


train_dataset = Data.TensorDataset(X_train, Y_train)  

train_loader = torch.utils.data.DataLoader(
                                            dataset=train_dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

class discharge_net(nn.Module):
    def __init__(self):
        super(discharge_net, self).__init__()
        self.linear = nn.Linear(INPUT_SIZE, INPUT_SIZE)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE,  
                            hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS,  
                            dropout=0.05,
                            batch_first=True,
                            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
                            )
        self.out = nn.Linear(HIDDEN_SIZE, OUT_SIZE)  

    def forward(self, X):

        l_out, (h_n, h_c) = self.lstm(X, None)  # out+hidden state ,None represents zero initial hidden state
        out = self.out(l_out[:, -1, :])
        return out

net = discharge_net()
print(net)

class Reluloss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self):
        super(Reluloss, self).__init__()

    def forward(self, input, target):
        mmm = input - target
        return nn.functional.relu(mmm)


optimizer = torch.optim.Adam(net.parameters(), lr=LR)
loss_func = Reluloss()  

total_step = len(train_loader)
Eopch = []
PICP_train_re_sby = []
PINAW_train_re_sby = []
PICP_test_re_sby = []
PINAW_test_re_sby = []

PICP_train_re_ghy = []
PINAW_train_re_ghy = []
PICP_test_re_ghy = []
PINAW_test_re_ghy = []

optEopch_sby = []
optPICP_train_sby = []
optPICP_test_sby = []
optPINAW_train_sby = []
optPINAW_test_sby = []

optEopch_ghy = []
optPICP_train_ghy = []
optPICP_test_ghy = []
optPINAW_train_ghy = []
optPINAW_test_ghy = []
losses = []

for k in range(1, repeatrun+1, 1):
    opt_test_NCWC_sby = 10000000
    opt_test_NCWC_ghy = 10000000
    for epoch in range(EPOCHES):  
        for i, (x, y) in enumerate(train_loader):  

            outputs = net(x)
            outputs = outputs.reshape(-1, OUT_SIZE)
            y = y.reshape(-1, 2)

            min_y0 = np.zeros(outputs[:, 0].shape)
            min_y = torch.from_numpy(min_y0).float()

            loss_sby_u = torch.mean((loss_func(y[:, 0].view(-1, 1), outputs[:, 1].view(-1, 1)))**2)
            loss_sby_l = torch.mean((loss_func(outputs[:, 0].view(-1, 1), y[:, 0].view(-1, 1)))**2)
            loss_sby_l2_u = torch.mean((loss_func(min_y.view(-1, 1), outputs[:, 1].view(-1, 1)))**2)
            loss_sby_l2_l = torch.mean((loss_func(min_y.view(-1, 1), outputs[:, 0].view(-1, 1))) ** 2)
            loss_sby_c = loss_sby_u + loss_sby_l
            loss_sby_phy = 1000 * loss_sby_l2_u + 1000 * loss_sby_l2_l

            loss_ghy_u = torch.mean((loss_func(y[:, 1].view(-1, 1), outputs[:, 3].view(-1, 1)))**2)
            loss_ghy_l = torch.mean((loss_func(outputs[:, 2].view(-1, 1), y[:, 1].view(-1, 1)))**2)
            loss_ghy_l2_u = torch.mean((loss_func(min_y.view(-1, 1), outputs[:, 3].view(-1, 1)))**2)
            loss_ghy_l2_l = torch.mean((loss_func(min_y.view(-1, 1), outputs[:, 2].view(-1, 1))) ** 2)
            loss_ghy_c = loss_ghy_u + loss_ghy_l
            loss_ghy_phy = 1000 * loss_ghy_l2_u + 1000 * loss_ghy_l2_l

            loss_sby_width = (abs(outputs[:, 1] - outputs[:, 0]))**2
            loss_sby_w = torch.mean(loss_sby_width)

            loss_ghy_width = (abs(outputs[:, 3] - outputs[:, 2]))**2
            loss_ghy_w = torch.mean(loss_ghy_width)

            delta_w1 = (PINC - PICP(outputs[:, 1].reshape(-1,1), outputs[:, 0].reshape(-1,1), np.array(y[:, 0].reshape(-1,1))))
            delta_w2 = (PINC - PICP(outputs[:, 3].reshape(-1,1), outputs[:, 2].reshape(-1,1), np.array(y[:, 1].reshape(-1, 1))))

            loss = (w1+delta_w1)*bei * loss_sby_c + (w2+delta_w2)*bei * loss_ghy_c + (w3-delta_w1)/bei * loss_sby_w + (w4-delta_w2)/bei * loss_ghy_w + loss_ghy_phy + loss_sby_phy

            optimizer.zero_grad()   
            loss.backward()         
            optimizer.step()        
            losses.append(loss.item())
            print('Epoch [{}/{}], step[{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCHES, i + 1, total_step, loss.item()))  

        with torch.no_grad():
            Y_pred = net(X_train)
            Y_pred_array = np.array(Y_pred.squeeze()).reshape(-1, OUT_SIZE)

            Y_valid_pred = net(X_valid)
            Y_valid_pred_array = np.array(Y_valid_pred.squeeze()).reshape(-1, OUT_SIZE)

            Y_test_pred = net(X_test)
            Y_test_pred_array = np.array(Y_test_pred.squeeze()).reshape(-1, OUT_SIZE)

            Y_pred_array_2 = Y_pred_array
            Y_test_pred_array_2 = Y_test_pred_array
            Y_valid_pred_array_2 = Y_valid_pred_array
            step1 = len(Y_pred_array_2)
            for i in range(step1):
                if Y_pred_array_2[i, 0] > Y_pred_array_2[i, 1]:
                    m1 = Y_pred_array_2[i, 0]
                    Y_pred_array_2[i, 0] = Y_pred_array_2[i, 1]
                    Y_pred_array_2[i, 1] = m1
                if Y_pred_array_2[i, 2] > Y_pred_array_2[i, 3]:
                    m2 = Y_pred_array_2[i, 2]
                    Y_pred_array_2[i, 2] = Y_pred_array_2[i, 3]
                    Y_pred_array_2[i, 3] = m2

            step3 = len(Y_valid_pred_array_2)
            for i in range(step3):
                if Y_valid_pred_array_2[i, 0] > Y_valid_pred_array_2[i, 1]:
                    nk2 = Y_valid_pred_array_2[i, 0]
                    Y_valid_pred_array_2[i, 0] = Y_valid_pred_array_2[i, 1]
                    Y_valid_pred_array_2[i, 1] = nk2
                if Y_valid_pred_array_2[i, 2] > Y_valid_pred_array_2[i, 3]:
                    nk1 = Y_valid_pred_array_2[i, 2]
                    Y_valid_pred_array_2[i, 2] = Y_valid_pred_array_2[i, 3]
                    Y_valid_pred_array_2[i, 3] = nk1

            step2 = len( Y_test_pred_array_2)
            for i in range(step2):
                if Y_test_pred_array_2[i, 0] > Y_test_pred_array_2[i, 1]:
                    n1 = Y_test_pred_array_2[i, 0]
                    Y_test_pred_array_2[i, 0] = Y_test_pred_array_2[i, 1]
                    Y_test_pred_array_2[i, 1] = n1
                if Y_test_pred_array_2[i, 2] > Y_test_pred_array_2[i, 3]:
                    n2 = Y_test_pred_array_2[i, 2]
                    Y_test_pred_array_2[i, 2] = Y_test_pred_array_2[i, 3]
                    Y_test_pred_array_2[i, 3] = n2

            Eopch.append(epoch)
            Y_train0 = np.array(Y_train[:, 0] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
            Y_valid0 = np.array(Y_valid[:, 0] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
            Y_test0 = np.array(Y_test[:, 0] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()

            Y_pred_array_20 = np.array(Y_pred_array_2[:, 0] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
            Y_pred_array_21 = np.array(Y_pred_array_2[:, 1] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
        
            Y_valid_pred_array_20 = np.array(Y_valid_pred_array_2[:, 0] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
            Y_valid_pred_array_21 = np.array(Y_valid_pred_array_2[:, 1] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
            
            Y_test_pred_array_20 = np.array(Y_test_pred_array_2[:, 0] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()
            Y_test_pred_array_21 = np.array(Y_test_pred_array_2[:, 1] * (sby_dis0_max - sby_dis0_min) + sby_dis0_min).flatten()

            Y_train1 = np.array(Y_train[:, 1] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
            Y_valid1 = np.array(Y_valid[:, 1] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
            Y_test1 = np.array(Y_test[:, 1] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()

            
            Y_pred_array_22 = np.array(Y_pred_array_2[:, 2] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
            Y_pred_array_23 = np.array(Y_pred_array_2[:, 3] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
            
            Y_valid_pred_array_22 = np.array(Y_valid_pred_array_2[:, 2] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
            Y_valid_pred_array_23 = np.array(Y_valid_pred_array_2[:, 3] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
           
            Y_test_pred_array_22 = np.array(Y_test_pred_array_2[:, 2] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()
            Y_test_pred_array_23 = np.array(Y_test_pred_array_2[:, 3] * (ghy_dis0_max - ghy_dis0_min) + ghy_dis0_min).flatten()

            
            PICP_train_sby = PICP(Y_pred_array_21.reshape((-1,1)), Y_pred_array_20.reshape((-1,1)), np.array(Y_train0.reshape(-1,1)))
            PICD_train_sby = PICD(Y_pred_array_21.reshape((-1, 1)), Y_pred_array_20.reshape((-1, 1)), np.array(Y_train0.reshape(-1, 1)))
            PIMSE_train_sby = PIMSE(Y_pred_array_21.reshape((-1, 1)), Y_pred_array_20.reshape((-1, 1)), np.array(Y_train0.reshape(-1, 1)))
            PINMSE_train_sby = PINMSE(Y_pred_array_21.reshape((-1, 1)), Y_pred_array_20.reshape((-1, 1)), np.array(Y_train0.reshape(-1, 1)))
            PINAW_train_sby = PINAW(Y_pred_array_21, Y_pred_array_20)
            PINRW_train_sby = PINRW(Y_pred_array_21, Y_pred_array_20)
            PIAW_train_sby = PIAW(Y_pred_array_21, Y_pred_array_20)

            
            PICP_valid_sby = PICP(Y_valid_pred_array_21.reshape((-1, 1)), Y_valid_pred_array_20.reshape((-1, 1)),
                                  np.array(Y_valid0).reshape(-1, 1))
            PICD_valid_sby = PICD(Y_valid_pred_array_21.reshape((-1, 1)), Y_valid_pred_array_20.reshape((-1, 1)),
                                  np.array(Y_valid0).reshape(-1, 1))
            PIMSE_valid_sby = PIMSE(Y_valid_pred_array_21.reshape((-1, 1)), Y_valid_pred_array_20.reshape((-1, 1)),
                                    np.array(Y_valid0).reshape(-1, 1))
            PINMSE_valid_sby = PINMSE(Y_valid_pred_array_21.reshape((-1, 1)), Y_valid_pred_array_20.reshape((-1, 1)),
                                      np.array(Y_valid0).reshape(-1, 1))
            PINAW_valid_sby = PINAW(Y_valid_pred_array_21, Y_valid_pred_array_20)
            PINRW_valid_sby = PINRW(Y_valid_pred_array_21, Y_valid_pred_array_20)
            PIAW_valid_sby = PIAW(Y_valid_pred_array_21, Y_valid_pred_array_20)

            
            PICP_test_sby = PICP(Y_test_pred_array_21.reshape((-1,1)), Y_test_pred_array_20.reshape((-1,1)), np.array(Y_test0).reshape(-1,1))
            PICD_test_sby = PICD(Y_test_pred_array_21.reshape((-1, 1)), Y_test_pred_array_20.reshape((-1, 1)),np.array(Y_test0).reshape(-1, 1))
            PIMSE_test_sby = PIMSE(Y_test_pred_array_21.reshape((-1, 1)), Y_test_pred_array_20.reshape((-1, 1)),np.array(Y_test0).reshape(-1, 1))
            PINMSE_test_sby = PINMSE(Y_test_pred_array_21.reshape((-1, 1)), Y_test_pred_array_20.reshape((-1, 1)), np.array(Y_test0).reshape(-1, 1))
            PINAW_test_sby = PINAW(Y_test_pred_array_21, Y_test_pred_array_20)
            PINRW_test_sby = PINRW(Y_test_pred_array_21, Y_test_pred_array_20)
            PIAW_test_sby = PIAW(Y_test_pred_array_21, Y_test_pred_array_20)

            
            NCWC_test_sby = NCWC(PINAW_test_sby, PICP_test_sby)
            NCWC_valid_sby = NCWC(PINAW_valid_sby, PICP_valid_sby)
            NCWC_train_sby = NCWC(PINAW_train_sby, PICP_train_sby)

            
            PICP_train_ghy = PICP(Y_pred_array_23.reshape((-1, 1)), Y_pred_array_22.reshape((-1,1)), np.array(Y_train1).reshape((-1,1)))
            PICD_train_ghy = PICD(Y_pred_array_23.reshape((-1, 1)), Y_pred_array_22.reshape((-1, 1)), np.array(Y_train1).reshape((-1, 1)))
            PIMSE_train_ghy = PIMSE(Y_pred_array_23.reshape((-1, 1)), Y_pred_array_22.reshape((-1, 1)), np.array(Y_train1).reshape((-1, 1)))
            PINMSE_train_ghy = PINMSE(Y_pred_array_23.reshape((-1, 1)), Y_pred_array_22.reshape((-1, 1)), np.array(Y_train1).reshape((-1, 1)))
            PINAW_train_ghy = PINAW(Y_pred_array_23, Y_pred_array_22)
            PINRW_train_ghy = PINRW(Y_pred_array_23, Y_pred_array_22)
            PIAW_train_ghy = PIAW(Y_pred_array_23, Y_pred_array_22)

            
            PICP_valid_ghy = PICP(Y_valid_pred_array_23.reshape((-1, 1)), Y_valid_pred_array_22.reshape((-1, 1)),
                                  np.array(Y_valid1).reshape((-1, 1)))
            PICD_valid_ghy = PICD(Y_valid_pred_array_23.reshape((-1, 1)), Y_valid_pred_array_22.reshape((-1, 1)),
                                  np.array(Y_valid1).reshape((-1, 1)))
            PIMSE_valid_ghy = PIMSE(Y_valid_pred_array_23.reshape((-1, 1)), Y_valid_pred_array_22.reshape((-1, 1)),
                                    np.array(Y_valid1).reshape((-1, 1)))
            PINMSE_valid_ghy = PINMSE(Y_valid_pred_array_23.reshape((-1, 1)), Y_valid_pred_array_22.reshape((-1, 1)),
                                      np.array(Y_valid1).reshape((-1, 1)))
            PINAW_valid_ghy = PINAW(Y_valid_pred_array_23, Y_valid_pred_array_22)
            PINRW_valid_ghy = PINRW(Y_valid_pred_array_23, Y_valid_pred_array_22)
            PIAW_valid_ghy = PIAW(Y_valid_pred_array_23, Y_valid_pred_array_22)

            
            PICP_test_ghy = PICP(Y_test_pred_array_23.reshape((-1,1)), Y_test_pred_array_22.reshape((-1,1)), np.array(Y_test1).reshape((-1,1)))
            PICD_test_ghy = PICD(Y_test_pred_array_23.reshape((-1, 1)), Y_test_pred_array_22.reshape((-1, 1)), np.array(Y_test1).reshape((-1, 1)))
            PIMSE_test_ghy = PIMSE(Y_test_pred_array_23.reshape((-1, 1)), Y_test_pred_array_22.reshape((-1, 1)), np.array(Y_test1).reshape((-1, 1)))
            PINMSE_test_ghy = PINMSE(Y_test_pred_array_23.reshape((-1, 1)), Y_test_pred_array_22.reshape((-1, 1)), np.array(Y_test1).reshape((-1, 1)))
            PINAW_test_ghy = PINAW(Y_test_pred_array_23, Y_test_pred_array_22)
            PINRW_test_ghy = PINRW(Y_test_pred_array_23, Y_test_pred_array_22)
            PIAW_test_ghy = PIAW(Y_test_pred_array_23, Y_test_pred_array_22)

            
            NCWC_test_ghy = NCWC(PINAW_test_ghy, PICP_test_ghy)
            NCWC_valid_ghy = NCWC(PINAW_valid_ghy, PICP_valid_ghy)
            NCWC_train_ghy = NCWC(PINAW_train_ghy, PICP_train_ghy)

            
            print('SBY Test:  ', 'PICP:{:.4f}，PINAW:{:.4f}，'.format(PICP_test_sby, PINAW_test_sby))
            print('SBY train:  ', 'PICP:{:.4f}，PINAW:{:.4f}，'.format(PICP_train_sby, PINAW_train_sby))
            print('ghy Test:  ','PICP:{:.4f}，PINAW:{:.4f}，'.format(PICP_test_ghy, PINAW_test_ghy))
            print('ghy train:  ','PICP:{:.4f}，PINAW:{:.4f}，'.format(PICP_train_ghy, PINAW_train_ghy))

            all_score_sby_train[epoch, 0] = NCWC_train_sby
            all_score_sby_train[epoch, 1] = PICP_train_sby
            all_score_sby_train[epoch, 2] = PICD_train_sby
            all_score_sby_train[epoch, 3] = PIMSE_train_sby
            all_score_sby_train[epoch, 4] = PINMSE_train_sby
            all_score_sby_train[epoch, 5] = PINAW_train_sby
            all_score_sby_train[epoch, 6] = PINRW_train_sby
            all_score_sby_train[epoch, 7] = PIAW_train_sby

            all_score_sby_valid[epoch, 0] = NCWC_valid_sby
            all_score_sby_valid[epoch, 1] = PICP_valid_sby
            all_score_sby_valid[epoch, 2] = PICD_valid_sby
            all_score_sby_valid[epoch, 3] = PIMSE_valid_sby
            all_score_sby_valid[epoch, 4] = PINMSE_valid_sby
            all_score_sby_valid[epoch, 5] = PINAW_valid_sby
            all_score_sby_valid[epoch, 6] = PINRW_valid_sby
            all_score_sby_valid[epoch, 7] = PIAW_valid_sby

            all_score_sby_test[epoch, 0] = NCWC_test_sby
            all_score_sby_test[epoch, 1] = PICP_test_sby
            all_score_sby_test[epoch, 2] = PICD_test_sby
            all_score_sby_test[epoch, 3] = PIMSE_test_sby
            all_score_sby_test[epoch, 4] = PINMSE_test_sby
            all_score_sby_test[epoch, 5] = PINAW_test_sby
            all_score_sby_test[epoch, 6] = PINRW_test_sby
            all_score_sby_test[epoch, 7] = PIAW_test_sby

            all_score_ghy_train[epoch, 0] = NCWC_train_ghy
            all_score_ghy_train[epoch, 1] = PICP_train_ghy
            all_score_ghy_train[epoch, 2] = PICD_train_ghy
            all_score_ghy_train[epoch, 3] = PIMSE_train_ghy
            all_score_ghy_train[epoch, 4] = PINMSE_train_ghy
            all_score_ghy_train[epoch, 5] = PINAW_train_ghy
            all_score_ghy_train[epoch, 6] = PINRW_train_ghy
            all_score_ghy_train[epoch, 7] = PIAW_train_ghy

            all_score_ghy_valid[epoch, 0] = NCWC_valid_ghy
            all_score_ghy_valid[epoch, 1] = PICP_valid_ghy
            all_score_ghy_valid[epoch, 2] = PICD_valid_ghy
            all_score_ghy_valid[epoch, 3] = PIMSE_valid_ghy
            all_score_ghy_valid[epoch, 4] = PINMSE_valid_ghy
            all_score_ghy_valid[epoch, 5] = PINAW_valid_ghy
            all_score_ghy_valid[epoch, 6] = PINRW_valid_ghy
            all_score_ghy_valid[epoch, 7] = PIAW_valid_ghy

            all_score_ghy_test[epoch, 0] = NCWC_test_ghy
            all_score_ghy_test[epoch, 1] = PICP_test_ghy
            all_score_ghy_test[epoch, 2] = PICD_test_ghy
            all_score_ghy_test[epoch, 3] = PIMSE_test_ghy
            all_score_ghy_test[epoch, 4] = PINMSE_test_ghy
            all_score_ghy_test[epoch, 5] = PINAW_test_ghy
            all_score_ghy_test[epoch, 6] = PINRW_test_ghy
            all_score_ghy_test[epoch, 7] = PIAW_test_ghy


        optEopch_re_sby = epoch
        opt_test_NCWC_sby = NCWC_test_sby
        optPICP_test_re_sby = PICP_test_sby
        optPICD_test_re_sby = PICD_test_sby
        optPIMSE_test_re_sby = PIMSE_test_sby
        optPINMSE_test_re_sby = PINMSE_test_sby
        optPINAW_test_re_sby = PINAW_test_sby
        optPINRW_test_re_sby = PINRW_test_sby
        optPIAW_test_re_sby = PIAW_test_sby

        opt_valid_NCWC_sby = NCWC_valid_sby
        optPICP_valid_re_sby = PICP_valid_sby
        optPICD_valid_re_sby = PICD_valid_sby
        optPIMSE_valid_re_sby = PIMSE_valid_sby
        optPINMSE_valid_re_sby = PINMSE_valid_sby
        optPINAW_valid_re_sby = PINAW_valid_sby
        optPINRW_valid_re_sby = PINRW_valid_sby
        optPIAW_valid_re_sby = PIAW_valid_sby

        opt_train_NCWC_sby = NCWC_train_sby
        optPICP_train_re_sby = PICP_train_sby
        optPICD_train_re_sby = PICD_train_sby
        optPIMSE_train_re_sby = PIMSE_train_sby
        optPINMSE_train_re_sby = PINMSE_train_sby
        optPINAW_train_re_sby = PINAW_train_sby
        optPINRW_train_re_sby = PINRW_train_sby
        optPIAW_train_re_sby = PIAW_train_sby

        Y_train01 = Y_train0
        Y_test01 = Y_test0
        Y_valid01 = Y_valid0
       
        Y_pred_array_201 = Y_pred_array_20
        Y_pred_array_211 = Y_pred_array_21
        
        Y_valid_pred_array_201 = Y_valid_pred_array_20
        Y_valid_pred_array_211 = Y_valid_pred_array_21
        
        Y_test_pred_array_201 = Y_test_pred_array_20
        Y_test_pred_array_211 = Y_test_pred_array_21
        print('SBY\n', 'optEopch_sby:{:.4f}'.format(optEopch_re_sby), '\n')
        print('SBY\n', 'SBY Test:  ', 'optPICP:{:.4f}，optPINAW:{:.4f}，'.format(optPICP_test_re_sby, optPIAW_test_re_sby))
        print('SBY\n', 'SBY valid: ', 'optPICP:{:.4f}，optPINAW:{:.4f}，'.format(optPICP_valid_re_sby, optPIAW_valid_re_sby))
        print('SBY\n', 'SBY train: ', 'optPICP:{:.4f}，optPINAW:{:.4f}，'.format(optPICP_train_re_sby, optPIAW_train_re_sby))
        save_name = f'opt_score_sby_{k}_'+'optmodel_params_discharge_total.pkl'
        torch.save(net.state_dict(), save_name)  

        optEopch_re_ghy = epoch
        opt_test_NCWC_ghy = NCWC_test_ghy
        optPICP_test_re_ghy = PICP_test_ghy
        optPICD_test_re_ghy = PICD_test_ghy
        optPIMSE_test_re_ghy = PIMSE_test_ghy
        optPINMSE_test_re_ghy = PINMSE_test_ghy
        optPINAW_test_re_ghy = PINAW_test_ghy
        optPINRW_test_re_ghy = PINRW_test_ghy
        optPIAW_test_re_ghy = PIAW_test_ghy

        opt_train_NCWC_ghy = NCWC_train_ghy
        optPICP_train_re_ghy = PICP_train_ghy
        optPICD_train_re_ghy = PICD_train_ghy
        optPIMSE_train_re_ghy = PIMSE_train_ghy
        optPINMSE_train_re_ghy = PINMSE_train_ghy
        optPINAW_train_re_ghy = PINAW_train_ghy
        optPINRW_train_re_ghy = PINRW_train_ghy
        optPIAW_train_re_ghy = PIAW_train_ghy

        opt_valid_NCWC_ghy = NCWC_valid_ghy
        optPICP_valid_re_ghy = PICP_valid_ghy
        optPICD_valid_re_ghy = PICD_valid_ghy
        optPIMSE_valid_re_ghy = PIMSE_valid_ghy
        optPINMSE_valid_re_ghy = PINMSE_valid_ghy
        optPINAW_valid_re_ghy = PINAW_valid_ghy
        optPINRW_valid_re_ghy = PINRW_valid_ghy
        optPIAW_valid_re_ghy = PIAW_valid_ghy

        
        Y_train11 = Y_train1
        Y_test11 = Y_test1
        Y_valid11 = Y_valid1
        
        Y_pred_array_221 = Y_pred_array_22
        Y_pred_array_231 = Y_pred_array_23
        
        Y_valid_pred_array_221 = Y_valid_pred_array_22
        Y_valid_pred_array_231 = Y_valid_pred_array_23
        
        Y_test_pred_array_221 = Y_test_pred_array_22
        Y_test_pred_array_231 = Y_test_pred_array_23
        print('ghy\n', 'optEopch_ghy:{:.4f}'.format(optEopch_re_ghy), '\n')
        print('ghy\n', 'ghy Test:  ', 'optPICP:{:.4f}，optPINAW:{:.4f}，'.format(optPICP_test_re_ghy, optPIAW_test_re_ghy))
        print('ghy\n', 'ghy valid: ','optPICP:{:.4f}，optPINAW:{:.4f}，'.format(optPICP_valid_re_ghy, optPIAW_valid_re_ghy))
        print('ghy\n', 'ghy train: ', 'optPICP:{:.4f}，optPINAW:{:.4f}，'.format(optPICP_train_re_ghy, optPIAW_train_re_ghy))
        save_name = f'opt_score_ghy_{k}_'+'optmodel_params_discharge_total.pkl'
        torch.save(net.state_dict(), save_name)  

        opt_score_sby[0, 0] = opt_train_NCWC_sby
        opt_score_sby[0, 1] = optPICP_train_re_sby
        opt_score_sby[0, 2] = optPICD_train_re_sby
        opt_score_sby[0, 3] = optPIMSE_train_re_sby
        opt_score_sby[0, 4] = optPINMSE_train_re_sby
        opt_score_sby[0, 5] = optPINAW_train_re_sby
        opt_score_sby[0, 6] = optPINRW_train_re_sby
        opt_score_sby[0, 7] = optPIAW_train_re_sby

        opt_score_sby[1, 0] = opt_valid_NCWC_sby
        opt_score_sby[1, 1] = optPICP_valid_re_sby
        opt_score_sby[1, 2] = optPICD_valid_re_sby
        opt_score_sby[1, 3] = optPIMSE_valid_re_sby
        opt_score_sby[1, 4] = optPINMSE_valid_re_sby
        opt_score_sby[1, 5] = optPINAW_valid_re_sby
        opt_score_sby[1, 6] = optPINRW_valid_re_sby
        opt_score_sby[1, 7] = optPIAW_valid_re_sby

        opt_score_sby[2, 0] = opt_test_NCWC_sby
        opt_score_sby[2, 1] = optPICP_test_re_sby
        opt_score_sby[2, 2] = optPICD_test_re_sby
        opt_score_sby[2, 3] = optPIMSE_test_re_sby
        opt_score_sby[2, 4] = optPINMSE_test_re_sby
        opt_score_sby[2, 5] = optPINAW_test_re_sby
        opt_score_sby[2, 6] = optPINRW_test_re_sby
        opt_score_sby[2, 7] = optPIAW_test_re_sby

        opt_score_ghy[0, 0] = opt_train_NCWC_ghy
        opt_score_ghy[0, 1] = optPICP_train_re_ghy
        opt_score_ghy[0, 2] = optPICD_train_re_ghy
        opt_score_ghy[0, 3] = optPIMSE_train_re_ghy
        opt_score_ghy[0, 4] = optPINMSE_train_re_ghy
        opt_score_ghy[0, 5] = optPINAW_train_re_ghy
        opt_score_ghy[0, 6] = optPINRW_train_re_ghy
        opt_score_ghy[0, 7] = optPIAW_train_re_ghy

        opt_score_ghy[1, 0] = opt_valid_NCWC_ghy
        opt_score_ghy[1, 1] = optPICP_valid_re_ghy
        opt_score_ghy[1, 2] = optPICD_valid_re_ghy
        opt_score_ghy[1, 3] = optPIMSE_valid_re_ghy
        opt_score_ghy[1, 4] = optPINMSE_valid_re_ghy
        opt_score_ghy[1, 5] = optPINAW_valid_re_ghy
        opt_score_ghy[1, 6] = optPINRW_valid_re_ghy
        opt_score_ghy[1, 7] = optPIAW_valid_re_ghy

        opt_score_ghy[2, 0] = opt_test_NCWC_ghy
        opt_score_ghy[2, 1] = optPICP_test_re_ghy
        opt_score_ghy[2, 2] = optPICD_test_re_ghy
        opt_score_ghy[2, 3] = optPIMSE_test_re_ghy
        opt_score_ghy[2, 4] = optPINMSE_test_re_ghy
        opt_score_ghy[2, 5] = optPINAW_test_re_ghy
        opt_score_ghy[2, 6] = optPINRW_test_re_ghy
        opt_score_ghy[2, 7] = optPIAW_test_re_ghy

        opt_re_y_boundary_train[:, 0] = Y_pred_array_201
        opt_re_y_boundary_train[:, 1] = Y_train01
        opt_re_y_boundary_train[:, 2] = Y_pred_array_211
        opt_re_y_boundary_train[:, 3] = Y_pred_array_221
        opt_re_y_boundary_train[:, 4] = Y_train11
        opt_re_y_boundary_train[:, 5] = Y_pred_array_231

        opt_re_y_boundary_valid[:, 0] = Y_valid_pred_array_201
        opt_re_y_boundary_valid[:, 1] = Y_valid01
        opt_re_y_boundary_valid[:, 2] = Y_valid_pred_array_211
        opt_re_y_boundary_valid[:, 3] = Y_valid_pred_array_221
        opt_re_y_boundary_valid[:, 4] = Y_valid11
        opt_re_y_boundary_valid[:, 5] = Y_valid_pred_array_231

        opt_re_y_boundary_test[:, 0] = Y_test_pred_array_201
        opt_re_y_boundary_test[:, 1] = Y_test01
        opt_re_y_boundary_test[:, 2] = Y_test_pred_array_211
        opt_re_y_boundary_test[:, 3] = Y_test_pred_array_221
        opt_re_y_boundary_test[:, 4] = Y_test11
        opt_re_y_boundary_test[:, 5] = Y_test_pred_array_231

    pd.DataFrame(opt_re_y_boundary_test).to_excel(f"opt_re_y_boundary_test_{k}.xlsx", header=['sby-lower-test', 'sby-observed-test', 'sby-upper-test', 'ghy-lower-test', 'ghy-observed-test', 'ghy-upper-test'], index=True)

    pd.DataFrame(opt_re_y_boundary_valid).to_excel(f"opt_re_y_boundary_valid{k}.xlsx",header=['sby-lower-valid', 'sby-observed-valid', 'sby-upper-valid', 'ghy-lower-valid', 'ghy-observed-valid', 'ghy-upper-valid'], index=True)
    pd.DataFrame(opt_re_y_boundary_train).to_excel(f"opt_re_y_boundary_train{k}.xlsx",header=['sby-lower-train', 'sby-observed-train', 'sby-upper-train', 'ghy-lower-train', 'ghy-observed-train', 'ghy-upper-train'], index=True)

    pd.DataFrame(opt_score_sby).to_excel(f"opt_score_sby_{k}.xlsx",
                                         header=['NCWC_sby-train-valid-test', 'PICP_sby-train-test',
                                                 'PICD_sby-train-test', 'PIMSE_sby-train-test', 'PINMSE_sby-train-test',
                                                 'PINAW_sby-train-test', 'PINRW_sby-train-test', 'PIAW_sby-train-test'],
                                         index=False)
    pd.DataFrame(opt_score_ghy).to_excel(f"opt_score_ghy_{k}.xlsx",
                                         header=['NCWC_ghy-train-valid-test', 'PICP_ghy-train-test',
                                                 'PICD_ghy-train-test', 'PIMSE_ghy-train-test', 'PINMSE_ghy-train-test',
                                                 'PINAW_ghy-train-test', 'PINRW_ghy-train-test', 'PIAW_ghy-train-test'],
                                         index=False)

    pd.DataFrame(all_score_sby_train).to_excel(f"all_score_sby_train_{k}.xlsx", header=['NCWC_sby-train','PICP_sby-train','PICD_sby-train', 'PIMSE_sby-train', 'PINMSE_sby-train', 'PINAW_sby-train', 'PINRW_sby-train','PIAW_sby-train'], index=False)
    pd.DataFrame(all_score_sby_valid).to_excel(f"all_score_sby_valid_{k}.xlsx", header=['NCWC_sby-valid','PICP_sby-valid','PICD_sby-valid', 'PIMSE_sby-valid', 'PINMSE_sby-valid', 'PINAW_sby-valid', 'PINRW_sby-valid','PIAW_sby-valid'], index=False)
    pd.DataFrame(all_score_sby_test).to_excel(f"all_score_sby_test_{k}.xlsx", header=['NCWC_sby-test','PICP_sby-test','PICD_sby-test', 'PIMSE_sby-test', 'PINMSE_sby-test', 'PINAW_sby-test', 'PINRW_sby-test','PIAW_sby-test'], index=False)

    pd.DataFrame(all_score_ghy_train).to_excel(f"all_score_ghy_train_{k}.xlsx", header=['NCWC_ghy-train','PICP_ghy-train','PICD_ghy-train', 'PIMSE_ghy-train','PINMSE_ghy-train', 'PINAW_ghy-train', 'PINRW_ghy-train','PIAW_ghy-train'], index=False)
    pd.DataFrame(all_score_ghy_valid).to_excel(f"all_score_ghy_valid_{k}.xlsx", header=['NCWC_ghy-valid','PICP_ghy-valid','PICD_ghy-valid', 'PIMSE_ghy-valid', 'PINMSE_ghy-valid', 'PINAW_ghy-valid', 'PINRW_ghy-valid','PIAW_ghy-valid'], index=False)
    pd.DataFrame(all_score_ghy_test).to_excel(f"all_score_ghy_test_{k}.xlsx", header=['NCWC_ghy-test','PICP_ghy-test','PICD_ghy-test', 'PIMSE_ghy-test', 'PINMSE_ghy-test', 'PINAW_ghy-test', 'PINRW_ghy-test','PIAW_ghy-test'], index=False)

    
    loss_data = pd.DataFrame(losses, columns=['Loss'])
    loss_data.to_excel(f"loss_data_{k}.xlsx", index=False)
  
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.4f}s'.format(time_elapsed // 60, time_elapsed % 60))


