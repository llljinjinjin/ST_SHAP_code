import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
from model.SWIN_trans import SwinTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold
import heapq
from einops import rearrange, repeat, reduce

device = torch.device('cuda')



def evaluate(model, x, y):
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=32, shuffle=True)
    num = 0
    model.eval()
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            inp, lab = d
            out = model(inp)
            _, pred = torch.max(out, 1)
            correct = torch.sum(pred == lab)
            num += correct.cpu().detach().numpy()
    return num * 1.0 / x.shape[0]


def eval_test(model, x, y):
    """
    :param model:
    :param x:
    :param y:
    :return: Return acc, confusion matrix, f1, kappa for test
    """
    data_set = TensorDataset(x, y)
    data_loader = DataLoader(dataset=data_set, batch_size=x.shape[0], shuffle=True)
    model.eval()
    acc, conMat, f, kappa = None, None, None, None
    with torch.no_grad():
        for i, d in enumerate(data_loader):
            inp, lab = d
            out = model(inp)
            _, pred = torch.max(out, 1)
            pred = pred.cpu().detach().numpy()
            lab = lab.cpu().detach().numpy()
            count=0
            for l in range(len(lab)):
                if lab[l]==0:
                    count=count+1
            acc = accuracy_score(pred, lab)  # acc
            conMat = confusion_matrix(pred, lab)  # confusion matrix
            f = f1_score(pred, lab, average=None)[0]  # f1
            # kappa
            axis1 = np.sum(conMat, axis=1)
            axis0 = np.sum(conMat, axis=0)
            p_e = np.sum(axis1 * axis0) / np.sum(conMat) ** 2
            p_o = acc
            kappa = (p_o - p_e) / (1 - p_e)
    return acc, conMat, f, kappa




def Split_Sets_10_Fold(total_fold, data):
    """
    :total_fold is the number of folds you set. I will take 10 with the argument here. data is the data I need to divide
    :train_index,test_index The index used to store train and test.

    """
    train_index = []
    test_index = []
    kf = KFold(n_splits=total_fold, shuffle=True, random_state=True)
    #Here set shuffle set to ture is to shuffle the order in the allocation
    for train_i, test_i in kf.split(data):
        train_index.append(train_i)
        test_index.append(test_i)
    return train_index, test_index

score = np.zeros([23,10,3])
f1 = np.zeros([23,10,3])
k = np.zeros([23,10,3])
conMat = np.zeros([23,10,3, 2, 2])
best_acc=np.zeros([23,3])
total_fold = 10

fea = np.load('../../data_input/dreamer_all/cnn_fea_map.npy').reshape([23,18*58,4,32,32])  # 23,1044,4,32,32
lab_a = np.load('../../data_input/dreamer_all/label_a.npy').reshape([23,18*58])  # 23,1044
lab_d = np.load('../../data_input/dreamer_all/label_d.npy').reshape([23,18*58])
lab_v = np.load('../../data_input/dreamer_all/label_v.npy').reshape([23,18*58])
print(fea.shape, lab_a.shape,lab_d.shape,lab_v.shape)
for ind in range(23):
    tmp_fea = fea[ind]  # (1044,4,32,32)  gamma
    [train_index, test_index] = Split_Sets_10_Fold(total_fold, tmp_fea)
    for shizhe in range(10):
        for biao in range(3):
            if biao==0:
                tmp_lab = lab_a[ind]  # (1044)
            elif biao==1:
                tmp_lab = lab_d[ind]  # (1044)
            else:
                tmp_lab = lab_v[ind]  # (1044)
            print(tmp_fea.shape, tmp_lab.shape)

            x_train = tmp_fea[train_index[shizhe]]  # Get training data 939,4,32,32
            x_test = tmp_fea[test_index[shizhe]]  # Get test data  105,4,32,32
            y_train = tmp_lab[train_index[shizhe]]  # Get training tag
            y_test = tmp_lab[test_index[shizhe]]  # Get test tag


            x_train_fea = torch.tensor(x_train, dtype=torch.float32)
            x_test_fea = torch.tensor(x_test, dtype=torch.float32)

            y_train_lab = torch.tensor(y_train, dtype=torch.long)
            y_test_lab = torch.tensor(y_test, dtype=torch.long)

            model = SwinTransformer(
                # Model parameter
                img_size=32,
                patch_size=4,
                in_chans=4, num_classes=2,
                embed_dim=96,
                depths=[2, 2, 6, 2],  # Number of layers
                num_heads=[3, 6, 12, 24],
                window_size=2,
                mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0.17, attn_drop_rate=0.17, drop_path_rate=0.1,
                ape=False, patch_norm=True,
                use_checkpoint=False, fused_window_process=False,

            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            loss_fun = torch.nn.CrossEntropyLoss().to(device)




            x_train_, y_train_ = x_train_fea.to(device), y_train_lab.to(device)  # xtrain(939,4,32,32)   ytrain(939)
            x_test_, y_test_ = x_test_fea.to(device), y_test_lab.to(device)  #xtest(105,4,32,32)      ytest(105)

            train_set = TensorDataset(x_train_, y_train_)
            train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

            train_loss_avg = []
            for epoch in tqdm(range(10)):
                train_loss = []
                model.train()
                for i, data in enumerate(train_loader):
                    inputs, labels = data
                    out = model(inputs)
                    loss = loss_fun(out, labels)
                    train_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                train_loss_avg.append(np.average(train_loss))
                print('\n avg loss: {}'.format(train_loss_avg[epoch]))

                train_acc = evaluate(model, x_train_, y_train_)
                print('\n Train acc: {}'.format(train_acc.item()))

            acc, Mat, f, kappa = eval_test(model, x_test_, y_test_)
            print('\n Test acc: {}'.format(acc))
            score[ind,shizhe,biao] = acc
            f1[ind,shizhe,biao] = f
            k[ind,shizhe,biao] = kappa
            conMat[ind,shizhe,biao] = Mat
            print("The {} category experiment acc of the {} person of the {} time is {}".format(biao,ind+1,shizhe+1,score[ind,shizhe,biao]))
            
            if(best_acc[ind,biao]<acc):
                best_acc[ind,biao]=acc
                torch.save(model, './model/dreamer/onlyswin_gamma_' + str(ind)+'_'+str(biao)+ '.pth')

    np.save(f'../../result/dreamer/acc{ind}.npy', score[ind])
    np.save(f'../../result/dreamer/f1{ind}.npy', f1[ind])
    np.save(f'../../result/dreamer/kappa{ind}.npy', k[ind])
    np.save(f'../../result/dreamer/conMat{ind}.npy', conMat[ind])

accsum = np.zeros([3, 23, 10])
for human in range(23):
    for sz in range(10):
        for zhi in range(3):
            if zhi == 0:
                print("The {} person's {} AROUSAL experiment acc is: {}".format(human + 1, sz + 1, score[sz, human, 0]))
                accsum[0, human, sz] = accsum[0, human, sz] + score[sz, human, 0]
            elif zhi == 1:
                print("The {} person's {} DOMINANCE experiment acc is: {}".format(human + 1, sz + 1, score[sz, human, 1]))
                accsum[1, human, sz] = accsum[1, human, sz] + score[sz, human, 1]
            else:
                print("The {} person's {} VALANCE experiment acc is: {}".format(human + 1, sz + 1, score[sz, human, 2]))
                accsum[2, human, sz] = accsum[2, human, sz] + score[sz, human, 2]


print("_____________________________________________________Ten percent off____________________________________________________________")
print("average Arousal：{}".format(np.mean(np.mean(accsum[0], axis=0))))
print("average Dominance：{}".format(np.mean(np.mean(accsum[1], axis=0))))
print("average Valance：{}".format(np.mean(np.mean(accsum[2], axis=0))))




