import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
from model.SWIN_trans import SwinTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import heapq
from einops import rearrange, repeat, reduce
import time

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
            acc = accuracy_score(pred, lab)  # acc
            conMat = confusion_matrix(pred, lab)  # confusion matrix
            # kappa
            axis1 = np.sum(conMat, axis=1)
            axis0 = np.sum(conMat, axis=0)
            p_e = np.sum(axis1 * axis0) / np.sum(conMat) ** 2
            p_o = acc
            kappa = (p_o - p_e) / (1 - p_e)
    return acc, conMat, kappa


score = np.zeros([15, 3])
k = np.zeros([15, 3])
conMat = np.zeros([15, 3, 3, 3])


for ind in range(1,16):
    for jnd in range(3):
        fea = np.load('./data_input/input_4/cnn_fea_map_{}.npy'.format(ind)).reshape([ 3, 15 * 57, 4, 32, 32])
        lab = np.load('./data_input/input_4/label_{}.npy'.format(ind)).reshape([3, 15 * 57])
        lab = lab + 1
        print(fea.shape, lab.shape)
        tmp_fea = fea[jnd]  # (855,4,32,32)  gamma
        tmp_lab = lab[jnd]  # (855)
        print(tmp_fea.shape, tmp_lab.shape)

        x_train, x_test, y_train, y_test = train_test_split(tmp_fea, tmp_lab, test_size=0.4, random_state=20)
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        x_train_fea = torch.tensor(x_train, dtype=torch.float32)
        x_test_fea = torch.tensor(x_test, dtype=torch.float32)
        y_train_lab = torch.tensor(y_train, dtype=torch.long)
        y_test_lab = torch.tensor(y_test, dtype=torch.long)

        model = SwinTransformer(
            # Model parameter
            img_size=32,
            patch_size=4,
            in_chans=4, num_classes=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],  # Number of floors
            num_heads=[3, 6, 12, 24],
            window_size=2,
            mlp_ratio=4., qkv_bias=True,
            drop_rate=0.17, attn_drop_rate=0.17, drop_path_rate=0.1,
            patch_norm=True,
            use_checkpoint=False,

        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        loss_fun = torch.nn.CrossEntropyLoss().to(device)




        x_train_, y_train_ = x_train_fea.to(device), y_train_lab.to(device)  # xtrain(513,8,32,32)   ytrain(513)
        x_test_, y_test_ = x_test_fea.to(device), y_test_lab.to(device)  #xtest(342,8,32,32)      ytest(342)

        train_set = TensorDataset(x_train_, y_train_)
        train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

        train_loss_avg = []
        for epoch in tqdm(range(500)):
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

        acc, Mat, kappa = eval_test(model, x_test_, y_test_)
        print('\n Test acc: {}'.format(acc))
        score[ind - 1, jnd] = acc
        k[ind - 1, jnd] = kappa
        conMat[ind - 1, jnd] = Mat
    torch.save(model, '../../model/seed/onlyswin_gamma_' + str(ind)+'_'+ str(jnd)+ '.pth')



accsum = np.zeros([15])
for human in range(1,16):
    for i in range(3):
        print("The {} person's {} experiment acc is: {}".format(human,i+1,score[human-1,i]))
        accsum[human-1] = accsum[human-1] + score[human-1,i]
    print("------------The average acc of the 3 experiments of {} person is {}-------------------".format(human,accsum[human-1]/3))

print(np.mean(np.mean(score, axis=0)))

np.save('./result/seed/acc.npy', score)
np.save('./result/seed/kappa.npy', k)
np.save('./result/seed/conMat.npy', conMat)

