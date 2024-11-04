import numpy.lib.format
import torch, torchvision
import shap
import numpy as np
from sklearn.model_selection import train_test_split  # Partition data set
from ST_SHAP_code.model.SWIN_trans import SwinTransformer
import heapq
# GradientExplainer Explain the model using the expected gradient (extension of the integrated gradient)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the models
model = torch.load('../model/seed/onlyswin_gamma_8.pth')

# Load data set
X= np.zeros([3, 342, 4, 32, 32])
y= np.zeros([3, 342])
for jnd in range(3):
    fea = np.load('../data_input/input_4/cnn_fea_map_8.npy').reshape([3, 15 * 57, 4, 32, 32])
    lab = np.load('../data_input/input_4/label_8.npy').reshape([3, 15 * 57])
    lab = lab + 1
    print(fea.shape, lab.shape)
    tmp_fea = fea[jnd]
    tmp_lab = lab[jnd]
    print(tmp_fea.shape, tmp_lab.shape)

    x_train, X_ONE, y_train, y_ONE = train_test_split(tmp_fea, tmp_lab, test_size=0.4, random_state=20)
    print(x_train.shape, X_ONE.shape, y_train.shape, y_ONE.shape)
    X[jnd] = X_ONE
    y[jnd] = y_ONE
X=X.reshape([3*342,4,32,32])
y=y.reshape([3*342])
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


X, y = X.to(device), y.to(device)
print(X.shape)  # test (342*3, 4, 32, 32)

# Select the sample you want to interpret
to_explain = X[:]
print("label:{}".format(y[:]))
print(to_explain.shape)  # test (342*3,4, 32, 32)

class_names = {'0': ['negative'], '1': ['neutral'], '2': ['positive']}

# print(model)
e = shap.GradientExplainer(model, X)

shap_values, indexes = e.shap_values(to_explain, ranked_outputs=1)
print(shap_values[0].shape)  # (342*3, 4, 32, 32)
print(indexes.shape)  # (342*3, 1)

to_explain = np.array(to_explain.cpu())

# plot the explanations swap the 2nd and 3rd dimensions
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
print(shap_values[0].shape)  # (342*3, 32, 32, 4)

indexes = np.array(indexes.cpu())

to_explain = to_explain.swapaxes(2, 3).swapaxes(-1, 1)
print(to_explain.shape)  # (342*3, 32, 32, 4)



shap_zero = np.zeros([np.sum(indexes == 0), 32, 32, 4])  # negative
shap_one = np.zeros([np.sum(indexes == 1), 32, 32, 4])  # neutral
shap_two = np.zeros([np.sum(indexes == 2), 32, 32, 4])  # positive

k, m, n = 0, 0, 0
for i in range(indexes.shape[0]):
    if indexes[i] == 0:
        shap_zero[k] = shap_values[0][i]
        k = k + 1
    elif indexes[i] == 1:
        shap_one[m] = shap_values[0][i]
        m = m + 1
    else:
        shap_two[n] = shap_values[0][i]
        n = n + 1
shap_zero = sum(shap_zero[:]) / len(shap_zero[:])
print(shap_zero.shape)
shap_one = sum(shap_one[:] / len(shap_one[:]))
shap_two = sum(shap_two[:] / len(shap_two[:]))

fin_explain = np.zeros([3, 32, 32, 4])
raw_0=[]
raw_1=[]
raw_2=[]
for i in range(to_explain.shape[0]):
    if y[i]==0:raw_0.append(to_explain[i])
    if y[i]==1:raw_1.append(to_explain[i])
    if y[i]==2:raw_2.append(to_explain[i])
raw_0=np.array(raw_0)
raw_1=np.array(raw_1)
raw_2=np.array(raw_2)
raw_0=np.mean(raw_0,axis=0)
raw_1=np.mean(raw_1,axis=0)
raw_2=np.mean(raw_2,axis=0)

fin_explain[0]=raw_0
fin_explain[1]=raw_1
fin_explain[2]=raw_2

fin_explain=np.mean(fin_explain,axis=3)
fin_explain=np.expand_dims(fin_explain,axis=3)

fin_shap_values = []
shap_values = np.zeros([3, 32, 32, 4])
shap_values[0] = shap_zero
shap_values[1] = shap_one
shap_values[2] = shap_two
fin_shap_values=shap_values

shap_values=np.mean(shap_values,axis=3)
fin_shap_values=np.expand_dims(shap_values,axis=3)
print(fin_explain.shape,fin_shap_values.shape)
# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][0])(torch.tensor([[0], [1], [2]], dtype=torch.long))
# print(index_names)

savedata0=np.zeros([2,32,32,1])
savedata0[0]=fin_explain[0]
savedata0[1]=fin_shap_values[0]
savneg=np.squeeze(savedata0)
np.save('./shapdatashow/seed/sub8/neg.npy',savneg)

savedata1=np.zeros([2,32,32,1])
savedata1[0]=fin_explain[1]
savedata1[1]=fin_shap_values[1]
savneu=np.squeeze(savedata1)
np.save('./shapdatashow/seed/sub8/neu.npy',savneu)

savedata2=np.zeros([2,32,32,1])
savedata2[0]=fin_explain[2]
savedata2[1]=fin_shap_values[2]
savpos=np.squeeze(savedata2)
np.save('./shapdatashow/seed/sub8/pos.npy',savpos)

# Generate image
shap.image_plot(fin_shap_values, fin_explain, index_names)
