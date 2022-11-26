# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:18:33 2021

@author: FromM
"""
import torch
import scipy.io
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import time
from Utils_ViscousDamping_Github import *
import os

if torch.cuda.is_available():
    cuda_tag = "cuda:0"
    device = torch.device(cuda_tag)  # ".to(device)" should be added to all models and inputs
    print("Running on " + cuda_tag)
else:
    device = torch.device("cpu")
    print("Running on the CPU")

writer = SummaryWriter("NN1")

start_time = time.time()
# =============================================================================
# fix random seed
# =============================================================================
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# Prepare data
# =============================================================================
data = scipy.io.loadmat(os.path.dirname(os.getcwd()) + '/ViscousDamping_Data_NoSource.mat')
t = data['t'].flatten()[:,None]
y = data['y'].flatten()[:,None]
dy = data['dy'].flatten()[:,None]
ddy = data['ddy'].flatten()[:,None]

# load test data
data_test = scipy.io.loadmat(os.path.dirname(os.getcwd()) + '/ViscousDamping_Data_NoSource_Test.mat')
y_test = data_test['y'].flatten()[:,None]
dy_test = data_test['dy'].flatten()[:,None]
ddy_test = data_test['ddy'].flatten()[:,None]
t_test = data_test['t'].flatten()[:,None]

# =============================================================================
# Design SNN
# =============================================================================
D_in_r = 5 # (1, y, dy, sign(y), sign(dy))
H = [2, 2] # hidden nodes for special ops
No_BiOp = [1, 1]
D_out_r = 1
model_r = SymbolicNet_PDENet(D_in_r, H, D_out_r, No_BiOp).to(device)

# =============================================================================
# pretrain SNN
# =============================================================================

Adam_epochs = 60000
model_r = AdamTrain(Adam_epochs, model_r, y, dy, ddy, y_test, dy_test, ddy_test, writer, device,
                    lr_model_r = 1e-3)

# # save model
torch.save({
            'model_state_dict': model_r.state_dict(),
            }, 'model_trained.tar')

# =============================================================================
# Sequential Threshold Training with adaptive tol.
# Alternate between thresholding and training until no SNN weights can be pruned by the specified tolerance
# =============================================================================

# backup model
with torch.no_grad():
    model_r_old = copy.deepcopy(model_r)

# set initial tol as 15 percentile of all SNN weights' absolute values
weights_SNN_list = [para.detach().to('cpu').numpy().flatten() for para in model_r.linear_weights_all]
weights_SNN = np.concatenate(weights_SNN_list)
My_Percentile_best = 5
My_Percentile_next = My_Percentile_best
tol_best = np.percentile(np.abs(weights_SNN), My_Percentile_best)
UpScale = 2

# record initial compound err
L0_coeff = 1e-1
ddy_test_pred, error_reg_best = ComputeAccuracy(model_r, y_test, dy_test, ddy_test, device, writer, PrintOption = False)
No_Nonzero_old = CountNonZeros(model_r)
No_Nonzero_best = No_Nonzero_old
err_total_old = error_reg_best + L0_coeff*No_Nonzero_best # this is similar to AIC

counter = 0

epoch_alt = 10 # alternating epochs in one IterativeSNNPrune
epochs_OneTime = 10000 # Adam training epochs in one IterativeSNNPrune

# heuristically adjust tol
for it_tol in range(25):

    weights_SNN_list = [para.detach().to('cpu').numpy().flatten() for para in model_r.linear_weights_all]
    weights_SNN = np.concatenate(weights_SNN_list)
    tol = np.percentile(np.abs(weights_SNN), My_Percentile_next)

    # alternation
    model_r, No_Nonzero_old, No_Nonzero_new, counter = IterativeSNNPruneTrain(epoch_alt, model_r, tol, y, dy, ddy, y_test, dy_test, ddy_test, writer, device, counter, epochs_OneTime)        

    # check if we should increase tol or not
    ddy_test_pred, error_reg_new = ComputeAccuracy(model_r, y_test, dy_test, ddy_test, device, writer, PrintOption = False)
    err_total_new = error_reg_new + L0_coeff*No_Nonzero_new # L0 coeff = 1e-2
    
    if err_total_new <= err_total_old:
        err_total_old = err_total_new
        My_Percentile_best = My_Percentile_next
        tol_best = tol
        
        if My_Percentile_best*UpScale < 100:
            My_Percentile_next = My_Percentile_best*UpScale
        else:
            My_Percentile_next = 99 # My_Percentile (max)
            
        # record loss 
        ddy_test_pred, error_reg_best = ComputeAccuracy(model_r, y_test, dy_test, ddy_test, device, writer, PrintOption = False)
        No_Nonzero_best = CountNonZeros(model_r)
                

    else:
        My_Percentile_next = My_Percentile_next*0.75
        UpScale = max([np.sqrt(UpScale), 1.2])

    # record loss 
    writer.add_scalar('No_Nonzero_best', L0_coeff*No_Nonzero_best, it_tol)
    writer.add_scalar('error_reg_best', error_reg_best, it_tol)
    writer.add_scalar('tol_best', tol_best, it_tol)
    writer.add_scalar('My_Percentile_best', My_Percentile_best, it_tol)

    # restore to the pretrained model. compare every tol on the same benchmark model. same idea as Pareto.
    with torch.no_grad():
        model_r = copy.deepcopy(model_r_old)
            
# best model
model_r, No_Nonzero_old, No_Nonzero_new, counter = IterativeSNNPruneTrain(epoch_alt, model_r, tol_best, y, dy, ddy, y_test, dy_test, ddy_test, writer, device, counter, epochs_OneTime)        
    
# save model
torch.save({
            'model_state_dict': model_r.state_dict(),
            }, 'model_trained_Pruned.tar')

DiscEq = SymbolicExpression(model_r)
print('The discovered Equation is ' + str(DiscEq))
writer.add_text('DiscEq', 'DiscEq:' + str(DiscEq))

# =============================================================================
# Results
# =============================================================================
elapsed = time.time() - start_time  
print('Training time: %.4f \n' % (elapsed))
writer.add_text('Time', 'Training time:' + str(elapsed))
 
ddy_pred, error_reg_new = ComputeAccuracy(model_r, y, dy, ddy, device, writer, PrintOption = True, NameTag = 'Train')
ddy_test_pred, error_reg_new = ComputeAccuracy(model_r, y_test, dy_test, ddy_test, device, writer, PrintOption = True, NameTag = 'Test')

# Add figures
Plotting(t, ddy_pred, ddy, writer, 'ddy_train')
Plotting(t, ddy_test_pred, ddy_test, writer, 'ddy_test')

# save data
scipy.io.savemat('PredSol.mat', {'ddy_pred': ddy_pred, 'ddy': ddy, 't': t,
                                 'ddy_test_pred': ddy_test_pred, 'ddy_test': ddy_test, 't_test': t_test})


writer.flush()
writer.close()
