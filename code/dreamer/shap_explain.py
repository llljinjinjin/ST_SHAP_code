import numpy.lib.format
import torch, torchvision
import shap
import numpy as np
from sklearn.model_selection import train_test_split  
from model.SWIN_trans import SwinTransformer
from group_lasso import GroupLasso
import heapq
# GradientExplainer explains the model using the expected gradient (extension of the integrated gradient)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the models
model = torch.load('../../model/dreamer/onlyswin_gamma_21_0.pth')  # 22_0 is the 23rd arousal

# Load data set
X= np.zeros([105, 4, 32, 32])
y= np.zeros([105])
fea = np.load('../../data_input/dreamer_all/cnn_fea_map.npy').reshape([23, 18 * 58, 4, 32, 32])
lab = np.load('../../data_input/dreamer_all/label_v.npy').reshape([23, 18 * 58])
tmp_fea=fea[21]  # Take the 22nd person data
tmp_lab=lab[21]
x_train, X_ONE, y_train, y_ONE = train_test_split(tmp_fea, tmp_lab, test_size=0.1, random_state=20)
print(X_ONE.shape,y_ONE.shape)
X = torch.tensor(X_ONE, dtype=torch.float32)  # 105,4,32,32
y = torch.tensor(y_ONE, dtype=torch.float32)  # 105
X, y = X.to(device), y.to(device)
print(X.shape)  # test(105, 4, 32, 32)


# Select the sample you want to interpret
to_explain = X[:]
print("label:{}".format(y[:]))
print(to_explain.shape)  # test(105,4, 32, 32)

class_names = {'0': ['LA'], '1': ['HA']}

# e for shap interpreter The first parameter can be a model object or a tuple (model, number of layers of the model) and the second parameter is the background data set
# If the input is a tuple, the returned shap value will be used for the input layer of the layer parameter must be a layer in the model
# print(model)
e = shap.GradientExplainer(model, X)

# Calculated shap_value The first parameter is the x-normalized image (the sample tensor used to interpret the model output) the second parameter is the number of selected output indexes and the third parameter is the number of samples taken
# shap_values is a numpy arraylist for each output column group
# indexes is a matrix that tells each sample which output indexes are selected as 'top'
# For a model with a single output, it returns a SHAP value tensor with the same shape as X; For a model with multiple outputs, it returns a list of SHAP value tensors, each with the same shape as X
# If ranked_outputs is None, then this tensor list matches the number of model outputs. If ranked_outputs is a positive integer that only explains many top-level model outputs, then return a pair (shap_values, indexes).
# Where shap_values is a list of tensors of length ranked_outputs and index is a matrix of which output indexes are selected as 'top' for each sample
shap_values, indexes = e.shap_values(to_explain, ranked_outputs=1, nsamples=32)  # The feature parameters are interpreted with an interpreter
print(shap_values[0].shape)  # (105, 4, 32, 32)
print(indexes.shape)  # (105, 1)

to_explain = np.array(to_explain.cpu())

# plot the explanations swap the 2nd and 3rd dimensions
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
print(shap_values[0].shape)  # (105, 32, 32, 4)

indexes = np.array(indexes.cpu())

to_explain = to_explain.swapaxes(2, 3).swapaxes(-1, 1)
print(to_explain.shape)  # (105, 32, 32, 4)

# print(np.sum(indexes == 2))

shap_zero = np.zeros([np.sum(indexes == 0), 32, 32, 4])  # L
shap_one = np.zeros([np.sum(indexes == 1), 32, 32, 4])  # H

k, m = 0, 0
for i in range(indexes.shape[0]):
    if indexes[i] == 0:
        shap_zero[k] = shap_values[0][i]
        k = k + 1
    else:
        shap_one[m] = shap_values[0][i]
        m = m + 1
shap_zero = sum(shap_zero[:]) / len(shap_zero[:])
print(shap_zero.shape)
shap_one = sum(shap_one[:] / len(shap_one[:]))

fin_explain = np.zeros([2, 32, 32, 4])
for i in range(2):
    fin_explain[i] = to_explain[list(y).index(i)]

fin_shap_values = []
shap_values = np.zeros([2, 32, 32, 4])
shap_values[0] = shap_zero
shap_values[1] = shap_one
fin_shap_values.append(shap_values)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][0])(torch.tensor([[0], [1]], dtype=torch.long))
# print(index_names)

# Generate image
shap.image_plot(fin_shap_values, fin_explain, index_names)
