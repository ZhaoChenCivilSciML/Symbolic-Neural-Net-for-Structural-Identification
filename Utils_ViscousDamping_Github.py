# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:30:26 2021

@author: FromM
"""
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
import sympy as sym

# =============================================================================
# Functions
# =============================================================================
def AdamTrain(Adam_epochs, model_r, y, dy, ddy, y_test, dy_test, ddy_test, writer, device,
              lr_model_r = 1e-3, counter = None):
    optimizer = torch.optim.Adam(model_r.linear_weights_all, lr_model_r)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5)

    loss_fn = torch.nn.MSELoss()
    y = torch.from_numpy(y).to(device).type(torch.float32)
    dy = torch.from_numpy(dy).to(device).type(torch.float32)
    ddy = torch.from_numpy(ddy).to(device).type(torch.float32)
    y_test = torch.from_numpy(y_test).to(device).type(torch.float32)
    dy_test = torch.from_numpy(dy_test).to(device).type(torch.float32)
    ddy_test = torch.from_numpy(ddy_test).to(device).type(torch.float32)

    for iter_Adam in tqdm(range(Adam_epochs)):
       
        # train loss
        ddy_pred = SNNpass(y, dy, model_r)
        loss = loss_fn(ddy_pred, ddy)

        # test loss
        ddy_test_pred = SNNpass(y_test, dy_test, model_r)
        loss_test = loss_fn(ddy_test_pred, ddy_test)
        
        if iter_Adam % 10 == 0:
            if counter == None:
                writer.add_scalar('loss_tr_Pre', loss.item(), iter_Adam)
                writer.add_scalar('loss_test_Pre', loss_test.item(), iter_Adam)
            else:
                writer.add_scalar('loss_tr_ADO', loss.item(), iter_Adam + counter*Adam_epochs)
                writer.add_scalar('loss_test_ADO', loss_test.item(), iter_Adam + counter*Adam_epochs)
            
        optimizer.zero_grad()
    
        loss.backward()
    
        optimizer.step()
        scheduler.step(loss_test)
    return model_r       

def ComputeAccuracy(model_r, y, dy, ddy, device, writer, PrintOption, NameTag = None):
    y = torch.from_numpy(y).to(device).type(torch.float32)
    dy = torch.from_numpy(dy).to(device).type(torch.float32)
    ddy_pred = SNNpass(y, dy, model_r).detach().to('cpu').numpy()

    error = np.linalg.norm(ddy_pred - ddy)/np.linalg.norm(ddy)*100
    
    if PrintOption:
        writer.add_text('Relative Error ' + NameTag, 'Relative Error(%):' + str(error))
        print(NameTag + ' Relative Error in percentage: %.4f \n' % (error))
    
    return ddy_pred, error

def SNNpass(y, dy, model_r):
    Phi = torch.cat((torch.ones_like(y), y, dy, torch.sign(y),
    torch.sign(dy), 
                     # torch.sin(dy)
                     ), 1)
    ddy_pred = model_r(Phi)
    return ddy_pred

def Plotting(t, x_full_pred, x_full, writer, nametag):
    fig = plt.figure()
    plt.plot(t, x_full_pred, label='pred')
    plt.plot(t, x_full, label='ref')
    plt.legend()
    writer.add_figure(nametag, fig)

def HardThreshold(model_r, SNN_threshold):   
    with torch.no_grad():
        # in case all weights are zeros in the end
        model_r_old = copy.deepcopy(model_r)
        
        # linear1
        # apply hard threshold to SNN weights
        ind1 = torch.abs(model_r.linear1.weight) < SNN_threshold
        model_r.linear1.weight[ind1] = 0
        # fix zero weights as zeros: Create Gradient mask
        gradient_mask1 = torch.ones_like(model_r.linear1.weight)
        gradient_mask1[ind1] = 0
        model_r.linear1.weight.register_hook(lambda grad: grad.mul_(gradient_mask1))
 
        # linear2
        # apply hard threshold to SNN weights
        ind2 = torch.abs(model_r.linear2.weight) < SNN_threshold
        model_r.linear2.weight[ind2] = 0
        # fix zero weights as zeros: Create Gradient mask
        gradient_mask2 = torch.ones_like(model_r.linear2.weight)
        gradient_mask2[ind2] = 0
        model_r.linear2.weight.register_hook(lambda grad: grad.mul_(gradient_mask2))

        # linear3
        # apply hard threshold to SNN weights
        ind3 = torch.abs(model_r.linear3.weight) < SNN_threshold
        model_r.linear3.weight[ind3] = 0
        # fix zero weights as zeros: Create Gradient mask
        gradient_mask3 = torch.ones_like(model_r.linear3.weight)
        gradient_mask3[ind3] = 0
        model_r.linear3.weight.register_hook(lambda grad: grad.mul_(gradient_mask3))
        
        # check if all weights are zeros
        if CountNonZeros(model_r) == 0:
            return model_r_old
    return model_r

def CountNonZeros(model_r):
    weights_SNN_list = [para.detach().to('cpu').numpy().flatten() for para in model_r.linear_weights_all]
    weights_SNN = np.concatenate(weights_SNN_list)
    No_Nonzero = np.count_nonzero(weights_SNN)
    return No_Nonzero

def SymbolicExpression(model_r):
    y, dy = sym.symbols('y, dy')
    # h = sym.Matrix([[1, y, dy, sym.functions.sign(dy), sym.sin(dy)]])
    h = sym.Matrix([[1, y, dy, sym.functions.sign(y),
    sym.functions.sign(dy)
    ]])
    # redundant PDE-Net SNN
    for count, para in enumerate(model_r.linear_weights_all):
        para = para.detach().to('cpu').numpy()
        h0 = copy.deepcopy(h)
        h = h@(para.T)*model_r.scale_fac
        if count == 0: 
            h_a = sym.zeros(h.shape[0], h.shape[1] - model_r.No_BiOp[count] + h0.shape[1])
            h_a[0, :h0.shape[1]] = h0[0, :h0.shape[1]]
            h_a[0, h0.shape[1]] = h[0, 0]*h[0, 1]
            h = h_a
        elif count == 1: 
            h_a = sym.zeros(h.shape[0], h.shape[1] - model_r.No_BiOp[count] + h0.shape[1])
            h_a[0, :h0.shape[1]] = h0[0, :h0.shape[1]]
            h_a[0, h0.shape[1]] = h[0, 0]*h[0, 1]
            h = h_a
    return sym.expand(h)

def IterativeSNNPruneTrain(epoch_alt, model_r, tol, y, dy, ddy, y_test, dy_test, ddy_test, writer, device, counter, epochs_OneTime):
    for it_alt in range(epoch_alt):
        # check if pruning is converged
        No_Nonzero_old = CountNonZeros(model_r)

        # prune SNN weights whose abs are smaller than tol. return the original model if all weights are pruned       
        model_r = HardThreshold(model_r, tol)
        
        # check if pruning is converged
        No_Nonzero_new = CountNonZeros(model_r)
        if No_Nonzero_old == No_Nonzero_new:
            break 
        else:
            No_Nonzero_old = No_Nonzero_new
            
        # refine SNN weights
        model_r = AdamTrain(epochs_OneTime, model_r, y, dy, ddy, y_test, dy_test, ddy_test, writer, device, lr_model_r = 1e-3, counter = counter)
        counter += 1
        
    return model_r, No_Nonzero_old, No_Nonzero_new, counter

# =============================================================================
# Classes
# =============================================================================
class SymbolicNet_PDENet(torch.nn.Module):
    def __init__(self, D_in_r, H, D_out_r, No_BiOp):
        super(SymbolicNet_PDENet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in_r, H[0], bias = False)
        self.linear2 = torch.nn.Linear(H[0] - No_BiOp[0] + D_in_r, H[1], bias = False) 
        self.linear3 = torch.nn.Linear(H[1] - No_BiOp[1] + H[0] - No_BiOp[0] + D_in_r, D_out_r, bias = False) 
        self.linear_weights_all = [self.linear1.weight, self.linear2.weight, self.linear3.weight]
        self.No_BiOp = No_BiOp
        self.H = H
        self.D_in_r = D_in_r

        self.scale_fac = 1
        
    def forward(self, X):       
        h1 = self.scale_fac*self.linear1(X)
        h1_a = torch.cat((X,
                          h1[:, 0:1]*h1[:, 1:2]
                          ), 1)
        
        h2 = self.scale_fac*self.linear2(h1_a)
        h2_a = torch.cat((h1_a,
                          h2[:, 0:1]*h2[:, 1:2], 
                          ), 1)

        h3 = self.scale_fac*self.linear3(h2_a)
        
        return h3
