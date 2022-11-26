# -*- coding: utf-8 -*-
"""
Othogonal matching pursuit
Created on Sat Mar 13 09:22:15 2021

@author: FromM
"""
import scipy.io
import numpy as np
import os
from sklearn.linear_model import OrthogonalMatchingPursuit
import matplotlib.pyplot as plt

# =============================================================================
# load data
# =============================================================================
# load train data
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
# build library
# =============================================================================
R_tr = np.concatenate((dy**2*y, dy**2*np.sign(dy), dy**2*np.sign(y), dy**2, dy*y**2, dy*y*np.sign(dy),
 dy*y*np.sign(y), dy*y,
 dy*np.sign(dy)*np.sign(y), dy*np.sign(dy), dy*np.sign(y), dy, y**3, y**2*np.sign(dy),
  y**2*np.sign(y), y**2, 
  y*np.sign(dy)*np.sign(y), y*np.sign(dy), y*np.sign(y), y,
  np.sign(dy)*np.sign(y), np.sign(dy), np.sign(y), np.ones_like(dy)), axis = 1)

R_test = np.concatenate((dy_test**2*y_test, dy_test**2*np.sign(dy_test), dy_test**2*np.sign(y_test), dy_test**2, dy_test*y_test**2, dy_test*y_test*np.sign(dy_test),
 dy_test*y_test*np.sign(y_test), dy_test*y_test,
 dy_test*np.sign(dy_test)*np.sign(y_test), dy_test*np.sign(dy_test), dy_test*np.sign(y_test), dy_test, y_test**3, y_test**2*np.sign(dy_test),
  y_test**2*np.sign(y_test), y_test**2, 
  y_test*np.sign(dy_test)*np.sign(y_test), y_test*np.sign(dy_test), y_test*np.sign(y_test), y_test,
  np.sign(dy_test)*np.sign(y_test), np.sign(dy_test), np.sign(y_test), np.ones_like(dy_test)),
                        axis = 1)

Lib_dict = ['dy**2*y', 'dy**2*np.sign(dy)', 'dy**2*np.sign(y)', 'dy**2', 'dy*y**2', 'dy*y*np.sign(dy)',
 'dy*y*np.sign(y)', 'dy*y',
 'dy*np.sign(dy)*np.sign(y)', 'dy*np.sign(dy)', 'dy*np.sign(y)', 'dy', 'y**3', 'y**2*np.sign(dy)',
  'y**2*np.sign(y)', 'y**2', 
  'y*np.sign(dy)*np.sign(y)', 'y*np.sign(dy)', 'y*np.sign(y)', 'y',
  'np.sign(dy)*np.sign(y)', 'np.sign(dy)', 'np.sign(y)', 'np.ones_like(dy)']

# =============================================================================
# othogonal matching pursuit
# =============================================================================
w_list = []
score_list = []
n_nonzero_coefs_list = []
err_list = []

for i_n_nonzero_coefs in range(1, int(R_tr.shape[1] + 1)):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs = i_n_nonzero_coefs)
    omp.fit(R_tr, ddy)
    score = omp.score(R_test, ddy_test)
    print('test R**2 score: ' + str(score))
    w = np.reshape(omp.coef_, [-1, 1])
    
    w_list.append(w)
    score_list.append(score)
    n_nonzero_coefs_list.append(i_n_nonzero_coefs)

    err = np.linalg.norm(ddy_test - R_test@w)/np.linalg.norm(ddy_test)*100
    err_list.append(err)

# =============================================================================
# select the best model
# =============================================================================
# pareto front plot
n_nonzero_coefs_all = np.stack(n_nonzero_coefs_list)
score_all = np.stack(score_list)
err_all = np.stack(err_list)

fig = plt.figure()
plt.plot(n_nonzero_coefs_all, score_all, 'o', color='black')

scipy.io.savemat('R_square.mat', {'n_nonzero_coefs_all':n_nonzero_coefs_all, 'score_all':score_all, 'err_all':err_all})

# =============================================================================
# express eq. run this part after pareto front
# =============================================================================
eq = []
w_best = w_list[1]
for i in range(w_best.shape[0]):
    if w_best[i, 0] != 0:
        eq.append(str(w_best[i, 0]) + '*' + Lib_dict[i])
        
print(eq)
