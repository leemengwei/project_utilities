import pandas as pd
import numpy as np
import os,time,sys
from IPython import embed
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
import copy

def show_me_input_data(pd_data):
    del(pd_data['Temp'])
    assert isinstance(pd_data, pd.core.frame.DataFrame), "I would like pandas with column names"
    pd_target_data = pd.read_csv("/mfs/home/limengwei/friction_compensation/data/planning_simulator.csv",sep=' ',index_col=None)[:-1][pd_data.columns].astype(float)
    data = pd_data.values
    target_data = pd_target_data.values
    embed()

    f1, ax1 = plt.subplots()
    f2, ax2 = plt.subplots()
    #Distribution:
    ax1.plot(data[::max(data.shape)//1000].T, c='r', linestyle='--', alpha=0.2)
    ax2.plot(target_data[::max(target_data.shape)//1000].T, c='b', linestyle='--', alpha=0.2)
    #Channel:
    ax1.plot(data.min(axis=0), 'k', label="train_min_max")
    ax1.plot(data.max(axis=0), 'k')
    ax2.plot(target_data.min(axis=0), c='k', label="simulator_min_max", alpha=0.8)
    ax2.plot(target_data.max(axis=0), c='k', alpha=0.8)
    #Misc:
    ax1.set_xticks(list(range(25-1)))   #Temp is del here.
    ax1.set_xticklabels(list(pd_data.columns), rotation=90, fontsize=5)
    ax2.set_xticks(list(range(25-1)))   #Temp is del here.
    ax2.set_xticklabels(list(pd_data.columns), rotation=90, fontsize=5)
    f1.legend()
    f2.legend()

    for idx,line in enumerate(ax1.lines): 
        line.set_color(plt.cm.Spectral_r(np.linspace(0,1,100))[idx])
    lineobj = plt.plot(data.T[:,:]) 
    plt.legend(iter(lineobj), list(range(25)))
    return

def performance_shape(raw_plan, inputs, model_part1):
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(212, projection='3d')
    #General:
    new = np.tile(inputs.detach().numpy().mean(axis=0), (1000,1))
    v_idx = 12
    p_idx = 13
    mannual_v = np.linspace(-inputs.detach().numpy().min(axis=0)[v_idx], inputs.detach().numpy().min(axis=0)[v_idx], 1000)
    mannual_p = np.linspace(-inputs.detach().numpy().min(axis=0)[p_idx], inputs.detach().numpy().min(axis=0)[p_idx], 1000)
    #F-v curve:
    new_v = copy.deepcopy(new)
    new_v[:,v_idx] = mannual_v
    output_v =  model_part1(torch.FloatTensor(new_v)).detach().numpy()
    ax1.plot(mannual_v, output_v)
    ax1.set_xlabel("Normalized speed")
    ax1.set_ylabel("Normalized compensation")
    ax1.set_title("v-F curve")
    #F-pos curve:
    new_p = copy.deepcopy(new)
    new_p[:,p_idx] = mannual_p
    output_p = model_part1(torch.FloatTensor(new_p)).detach().numpy()
    ax2.plot(mannual_p, output_p)
    ax2.set_xlabel("Normalized position")
    ax2.set_ylabel("Normalized compensation")
    ax2.set_title("p-F curve")
    #F-v-pos surface:
    new_vp = np.tile(new, (1000,1))
    new_vp[:,p_idx] = np.random.uniform(-inputs.detach().numpy().min(axis=0)[v_idx], inputs.detach().numpy().min(axis=0)[v_idx], 1000*1000)
    new_vp[:,v_idx] = np.random.uniform(-inputs.detach().numpy().min(axis=0)[p_idx], inputs.detach().numpy().min(axis=0)[p_idx], 1000*1000)
    output_vp = model_part1(torch.FloatTensor(new_vp)).detach().numpy()
    ax3.scatter3D(new_vp[:,v_idx][::100], new_vp[:,p_idx][::100], output_vp.flatten()[::100], s=0.5, alpha=0.5, c=output_vp.flatten()[::100], cmap='Spectral_r')
    ax3.set_xlabel("Normalized speed")
    ax3.set_ylabel("Normalized position")
    ax3.set_zlabel("Normalized compensation")
    ax3.set_title("v-p-F surface")
    plt.show()

    sys.exit()
    return



if __name__ == "__main__":
    print("NN input Analyzer")
    print("Call me in your scripts...")

