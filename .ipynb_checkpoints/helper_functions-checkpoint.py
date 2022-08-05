import os
import pandas as pd
import pickle
import numpy as np
from brainspace.gradient import ProcrustesAlignment
import nibabel as nib
import matplotlib.pyplot as plt
import random
from neuroCombat import neuroCombat as NC

random.seed(0)

def get_gradients(grad_folder_path, grad_temp_path, pt_info_path):
    charsplit = '_'
    pt_info = pd.read_csv(pt_info_path)
    
    pts = os.listdir(grad_folder)
    pts = [x for x in pts if x[0] != charsplit]
    pts = [x for x in pts if x.split(charsplit)[0] in list(pt_info.ID)]
    
    with open(grad_temp, 'rb') as f:
        temp = pickle.load(f)
    procrustes_template = temp['gradient']
    
    ## align gradients procrustes
    all_grad = []
    all_eig = []
    
    for pt in pts:
        with open('%s/%s'%(grad_folder, pt), 'rb') as f:
            temp = pickle.load(f)
        all_grad.append(temp['gradient'])
        all_eig.append(temp['lambda'])
    
    align = ProcrustesAlignment()
    all_grad = align.fit(all_grad, reference=procrustes_template).aligned_
    
    prot = []
    lat = []
    
    for i, pt in enumerate(pts):
        pt_name = pt.split(charsplit)[0]
        
        prot.append(pt_info[pt_info.ID == pt_name.split(charsplit)[0]]['3T_Protocol'].values[0])
        
        if 'Control' in pt_info[pt_info.ID == pt_name]['Lateralization'].values[0]:
            lat.append(0)
        elif 'L' in pt_info[pt_info.ID == pt_name]['Lateralization'].values[0]:
            lat.append(1)
        else:
            lat.append(2)
    
    ## harmonize data using neurocombat (regress out differences due to fMRI protocol)
    
    all_grad = np.dstack(all_grad)
    data = np.vstack([all_grad[:,0,:], all_grad[:,1,:]])
    
    covars = {'batch': prot,
          'lat':lat} 
    covars = pd.DataFrame(covars)  
    
    categorical_cols = ['lat']
    batch_col = 'batch'
    
    data_combat = NC(dat=data,
        covars=covars,
        batch_col=batch_col,
        categorical_cols=categorical_cols)["data"]
    
    out = np.dstack([data_combat[:all_grad.shape[0],:], data_combat[all_grad.shape[0]:,:]])
    
    hc_grad = []
    ep_L_grad = []
    ep_R_grad = []
    
    for i, pt in enumerate(pts):  
        pt_name = pt.split(charsplit)[0]

        if 'Control' in pt_info[pt_info.ID == pt_name]['Lateralization'].values[0]:
            hc_grad.append(out[:,i,:])
        elif 'L' in pt_info[pt_info.ID == pt_name]['Lateralization'].values[0]:
            ep_L_grad.append(out[:,i,:])
        else:
            ep_R_grad.append(out[:,i,:])
    
    hc_grad = np.dstack(hc_grad)
    ep_L_grad = np.dstack(ep_L_grad)
    ep_R_grad = np.dstack(ep_R_grad)
    
    return (hc_grad, ep_L_grad, ep_R_grad)

def plot_avg_grad(hc_grad, ep_L_grad, ep_R_grad, roi):
    
    min1 = np.min([hc_grad[:,1], ep_L_grad[:,1], ep_R_grad[:,1]])
    max1 = np.max([hc_grad[:,1], ep_L_grad[:,1], ep_R_grad[:,1]])
    min0 = np.min([hc_grad[:,0], ep_L_grad[:,0], ep_R_grad[:,0]])
    max0 = np.max([hc_grad[:,0], ep_L_grad[:,0], ep_R_grad[:,0]])
    
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].scatter(hc_grad[:,1], hc_grad[:,0], s=2, c=roi)
    axs[0].set_title('Healthy')
    axs[0].set_xlabel('Gradient 2')
    axs[0].set_ylabel('Gradient 1')
    axs[0].set_xlim([min1,max1])
    axs[0].set_ylim([min0,max0])
    axs[1].scatter(ep_L_grad[:,1], ep_L_grad[:,0], s=2, c=roi)
    axs[1].set_title('Left')
    axs[1].set_xlim([min1,max1])
    axs[1].set_ylim([min0,max0])
    axs[2].scatter(ep_R_grad[:,1], ep_R_grad[:,0], s=2, c=roi)
    axs[2].set_title('Right')
    axs[2].set_xlim([min1,max1])
    axs[2].set_ylim([min0,max0])
    plt.show()
    
    return None

def bhatta(a, b):
    m0 = np.mean(a,axis=0)
    m1 = np.mean(b,axis=0)
    diff = m1 - m0
    
    S0 = np.cov(a, rowvar=False)
    S1 = np.cov(b, rowvar=False)
    
    S_avg = (S0 + S1)/2
    iS_avg = np.linalg.inv(S_avg)
    
    det_term = .5 * np.log(np.linalg.det(S_avg) / np.sqrt(np.linalg.det(S0) * np.linalg.det(S1)))
    mahal_term = .125 * diff.T @ iS_avg @ diff
    
    return det_term + mahal_term

def permute_groups(ep_L_grad, ep_R_grad):
    
    ep_L_size = ep_L_grad.shape[2]
    
    all_grad = np.dstack([ep_L_grad, ep_R_grad])
    
    pts = np.arange(all_grad.shape[2])
    random.shuffle(pts)
    
    ep_L_shuffle = all_grad[:,:,pts[:ep_L_size]]
    ep_R_shuffle = all_grad[:,:,pts[ep_L_size:]]
    
    return (ep_L_shuffle, ep_R_shuffle)