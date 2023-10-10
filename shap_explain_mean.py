import numpy.lib.format
import torch, torchvision
import shap
import numpy as np
from sklearn.model_selection import train_test_split  # Partition data set
from model.SWIN_trans import SwinTransformer
import heapq
# GradientExplainer Explain the model using the expected gradient (extension of the integrated gradient)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the models
model = torch.load('./model/seed/onlyswin_1.pth')

# Load data set
X= np.zeros([3, 342, 4, 32, 32])
y= np.zeros([3, 342])
for jnd in range(3):
    fea = np.load('./data_input/input_4/cnn_fea_map_1.npy').reshape([3, 15 * 57, 4, 32, 32])
    lab = np.load('./data_input/input_4/label_1.npy').reshape([3, 15 * 57])
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

# The first parameter of the shap interpreter can be a model object or a tuple (the model, the number of layers of the model) and the second parameter is the background data set
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
print(shap_values[0].shape)  # (342*3, 4, 32, 32)
print(indexes.shape)  # (342*3, 1)

to_explain = np.array(to_explain.cpu())

# plot the explanations swap the 2nd and 3rd dimensions
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
print(shap_values[0].shape)  # (342*3, 32, 32, 4)

indexes = np.array(indexes.cpu())

to_explain = to_explain.swapaxes(2, 3).swapaxes(-1, 1)
print(to_explain.shape)  # (342*3, 32, 32, 4)

# print(np.sum(indexes == 2))

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
for i in range(3):
    fin_explain[i] = to_explain[list(y).index(i)]

fin_shap_values = []
shap_values = np.zeros([3, 32, 32, 4])
shap_values[0] = shap_zero
shap_values[1] = shap_one
shap_values[2] = shap_two
fin_shap_values.append(shap_values)

# get the names for the classes
index_names = np.vectorize(lambda x: class_names[str(x)][0])(torch.tensor([[0], [1], [2]], dtype=torch.long))
# print(index_names)

# Generate image
shap.image_plot(fin_shap_values, fin_explain, index_names)
