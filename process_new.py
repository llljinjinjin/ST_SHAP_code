import scipy.signal
import os
import re
import numpy as np
import scipy.io
from scipy.signal import butter, lfilter
from scipy import interpolate
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import mne
from matplotlib import pyplot as plt
from einops import rearrange, repeat, reduce
# Read data
def read_data(num):
    """
    :return: res: data, lab: label
    """
    res = []  # Each person has 3 sessions, each session contains 15 experiments, and each experiment takes 57 time periods (3, 15, 57, 62, 800).
    l = []  # (3, 15, 57)
    files = os.listdir('./data_input/dataset')
    label = scipy.io.loadmat('./data_input/label/label.mat')['label'][0]
    for index, file in enumerate(files):
        if not re.search('[0-9].*.mat', file):
            continue

        number = int(file.split('_')[0])
        if number == num:
            sig = np.zeros([15, 57, 62, 800])  #  Take 1 minute in the middle of 4 minutes, and then 4 seconds as the window 200*4=800, divided into 57 segments.
            lab = np.zeros([15, 57])
            data_mat = scipy.io.loadmat('../../data_input/dataset/' + file)
            del data_mat['__header__']
            del data_mat['__version__']
            del data_mat['__globals__']
            for i, (k, v) in enumerate(data_mat.items()):
                tmp = v[:, 180 * 100: 300 * 100]  # 4min  1.5min-2.5min
                for j in range(57):
                    sig[i, j] = tmp[:, j * 200:(j + 4) * 200]
                    lab[i, j] = label[i]
            res.append(sig)
            l.append(lab)
    return res, l


def data_1Dto2D(data, Y=9, X=9):  # 9 x 9 network topology
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, data[0], data[1], data[2], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[3], 0, data[4], 0, 0, 0)
    data_2D[2] = (data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13])
    data_2D[3] = (data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[21], data[22])
    data_2D[4] = (data[23], data[24], data[25], data[26], data[27], data[28], data[29], data[30], data[31])
    data_2D[5] = (data[32], data[33], data[34], data[35], data[36], data[37], data[38], data[39], data[40])
    data_2D[6] = (data[41], data[42], data[43], data[44], data[45], data[46], data[47], data[48], data[49])
    data_2D[7] = (0, data[50], data[51], data[52], data[53], data[54], data[55], data[56], 0)
    data_2D[8] = (0, 0, data[57], data[58], data[59], data[60], data[61], 0, 0)
    return data_2D


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


def calculate_differential_entropy(data):
    return 0.5 * np.log(2 * np.pi * np.exp(1) * np.var(data) + np.array(1e-5))


def read(num):
    res = np.zeros([3, 15, 57, 62, 800])
    lab = np.zeros([3, 15, 57])
    feature, label = read_data(num)
    for j in range(3):
        res[j] = feature[j]
        lab[j] = label[j]


    res_band = np.zeros([3, 15, 57, 62, 800])    #Gamma-band filtering
    for j in range(3):
        for v in range(15):
            for m in range(57):
                for k in range(62):
                    res_band[j, v, m, k] = butter_bandpass_filter(res[j, v, m, k], 31, 50, 200)



    entropy = np.zeros([ 3, 15, 57, 62, 4])       #This one is divided into four time periods, calculates differential entropy once per second, but has not been topologically mapped
    for m in range(3):
        for n in range(15):
            for o in range(57):
                for k in range(62):
                    for v in range(4):
                        entropy[ m, n, o, k, v] = calculate_differential_entropy(res_band[m, n, o, k, v * 200:(v + 1) * 200])


    fea = np.zeros([ 3, 15, 57, 4, 32, 32])       #Topology mapping and interpolation
    for m in range(3):
        for n in range(15):
            for v in range(57):
                for k in range(4):
                    mapped = data_1Dto2D(entropy[ m, n, v, :, k])
                    interp = interpolate.interp2d(np.linspace(0, 8, 9), np.linspace(0, 8, 9), mapped, kind='cubic')
                    x2 = np.linspace(0, 8, 32)  # A number of equally spaced numbers is generated within the specified interval
                    y2 = np.linspace(0, 8, 32)
                    Z2 = interp(x2, y2)
                    fea[m, n, v, k, :, :] = Z2

    print('\n ########## save ############')


    np.save('../../data_input/input_4/cnn_fea_map_{}.npy'.format(num), fea)  # (3,15,57,4,32,32)
    np.save('../../data_input/input_4/label_{}.npy'.format(num), lab)  #  (3,15,57)


for i in tqdm(range(1, 16)):
    read(i)
