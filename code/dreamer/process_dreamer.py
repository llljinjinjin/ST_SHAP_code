import scipy.signal
import os
import re
import numpy as np
import scipy.io
from scipy.signal import butter, lfilter
from scipy import interpolate
from tqdm import tqdm
from einops import rearrange, repeat
from matplotlib import pyplot as plt
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def data_1Dto2D(data, Y=9, X=9):  # 9 x 9 network topology
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[0], 0, data[1], 0, 0, 0)
    data_2D[2] = (data[2], 0, data[3], 0, 0, 0, data[4], 0, data[5])
    data_2D[3] = (0, data[6], 0, 0, 0, 0, 0, data[7], 0)
    data_2D[4] = (data[8], 0, 0, 0, 0, 0, 0, 0, data[9])
    data_2D[5] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[6] = (data[10], 0, 0, 0, 0, 0, 0, 0, data[11])
    data_2D[7] = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[12], 0, data[13], 0, 0, 0)
    return data_2D

def calculate_differential_entropy(data):
    return 0.5 * np.log(2 * np.pi * np.exp(1) * np.var(data) + np.array(1e-5))


dreamer = scipy.io.loadmat('../../data_input/DREAMER.mat')
del dreamer['__header__']
del dreamer['__version__']
del dreamer['__globals__']
dreamer=dreamer['DREAMER']['Data'][0][0][0]

data_eeg=np.zeros([23,18,14,7808])
label_Valence=np.zeros([23,18])
label_Arousal=np.zeros([23,18])
label_Dominance=np.zeros([23,18])

sig = np.zeros([23, 18,58,14, 512])  # 61s, then 4 seconds for the window 128*4=512, divided into 58 segments.
lab_v = np.zeros([23, 18,58])
lab_a = np.zeros([23, 18,58])
lab_d = np.zeros([23, 18,58])
for i in range(23):
    tmp_=np.zeros([18,7808,14])
    list_data = dreamer[i][0][0]
    data_eegtmp=list_data[2][0][0][0]# 18,1
    label_Valence[i]=list_data[4].reshape([18])  #18
    label_Arousal[i]=list_data[5].reshape([18])  #18
    label_Dominance[i]=list_data[6].reshape([18])  #18
    for film in range(18):
        tmp_[film]=data_eegtmp[film][0]  #18,7808,14
        if label_Valence[i][film] <= 3:
            label_Valence[i][film] = 0
        elif label_Valence[i][film] >3:
            label_Valence[i][film] = 1

        if label_Arousal[i][film]<=3:
            label_Arousal[i][film]=0
        elif label_Arousal[i][film]>3:
            label_Arousal[i][film]=1

        if label_Dominance[i][film]<=3:
            label_Dominance[i][film]=0
        elif label_Dominance[i][film]>3:
            label_Dominance[i][film]=1

    data_eeg[i] = rearrange(tmp_, 'h w c -> h c w')  # 18,14,7808


    tmp=data_eeg[i]  #18,14,7808
    for f in range(18):
        for j in range(58):
            sig[i,f, j] = tmp[f,:, j * 128:(j + 4) * 128]
            lab_v[i,f, j] = label_Valence[i][f]
            lab_a[i,f ,j] = label_Arousal[i][f]
            lab_d[i,f, j] = label_Dominance[i][f]



res_band = np.zeros([23, 18,58, 14, 512])    #Gamma-band filtering
for j in range(23):
    for k in range(18):
        for v in range(58):
            for m in range(14):
                res_band[j, k,v, m] = butter_bandpass_filter(sig[j,k, v, m], 31, 50, 128)

entropy = np.zeros([ 23, 18,58, 14, 4])       #This one is divided into four time periods, calculates differential entropy once per second, but has not been topologically mapped
for m in range(23):
    for k in range(18):
        for n in range(58):
            for o in range(14):
                for v in range(4):
                    entropy[ m,k, n, o, v] = calculate_differential_entropy(res_band[m, k,n, o,v * 128:(v + 1) * 128])

fea = np.zeros([ 23,18, 58, 4, 32, 32])       #Topology mapping and interpolation
for m in range(23):
    for h in range(18):
        for v in range(58):
            for k in range(4):
                mapped = data_1Dto2D(entropy[ m,h,v, :, k])
                interp = interpolate.interp2d(np.linspace(0, 8, 9), np.linspace(0, 8, 9), mapped, kind='cubic')
                x2 = np.linspace(0, 8, 32)
                y2 = np.linspace(0, 8, 32)
                Z2 = interp(x2, y2)
                fea[ m, h,v, k, :, :] = Z2

print('\n ########## save ############')
np.save('../../data_input/dreamer_all/cnn_fea_map.npy',fea)  # (23,18,58,4,32,32)
np.save('../../data_input/dreamer_all/label_a.npy',lab_a)  #  (23,18,58)
np.save('../../data_input/dreamer_all/label_v.npy',lab_v)  #  (23,18,58)
np.save('../../data_input/dreamer_all/label_d.npy',lab_d)  #  (23,18,58)
