import numpy.lib.format
import torch, torchvision
import shap
import numpy as np
from sklearn.model_selection import train_test_split  # 划分数据集
from model.swin_transformer import SwinTransformer
from group_lasso import GroupLasso
import heapq
# GradientExplainer 解释使用预期梯度的模型(集成梯度的扩展)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load the models
# 加载模型 评估模型
model = torch.load('./model/seed/onlyswin_1.pth') 

# 加载数据集 seed脑电数据集
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
print(X.shape)  # 测试部分(342*3, 4, 32, 32)

# 选定要解释的样本
to_explain = X[:]
print("label:{}".format(y[:]))
print(to_explain.shape)  # 测试部分(342*3,4, 32, 32)

class_names = {'0': ['negative'], '1': ['neutral'], '2': ['positive']}

# e为shap解释器 第一个参数可以为模型对象或者一个元组(模型, 模型的层数) 第二个参数为背景数据集
# 如果输入是元组, 则返回的shap值将用于层参数的输入 层必须是模型中的一个层
# print(model)
e = shap.GradientExplainer(model, X)

# 计算的shap_value值 第一个参数为X 归一化后的图像(用于解释模型输出的样本张量) 第二个参数为选定输出索引的数量 第三个参数为采用的样本数
# shap_values是每个输出列组的numpy数组列表
# indexes是一个矩阵, 告诉每个样本哪些输出索引被选为'top'
# 对于具有单个输出的模型, 它返回一个形状与X相同的SHAP值张量; 对于具有多个输出的模型来说, 它返回SHAP值张量列表, 每个张量的形状都与X相同
# 如果ranked_outputs为None, 则此张量列表与模型输出的数量匹配 如果ranked_outputs是一个正整数, 只解释许多顶级模型输出，则返回一对(shap_values, indexes)
# 其中shap_values是一个长度为ranked_outputs的张量列表, index是一个矩阵, 每个样本哪些输出索引被选为'top'
shap_values, indexes = e.shap_values(to_explain, ranked_outputs=1, nsamples=32)  # 用解释器对特征参数进行解释
print(shap_values[0].shape)  # (342*3, 4, 32, 32)
print(indexes.shape)  # (342*3, 1)

to_explain = np.array(to_explain.cpu())

# plot the explanations 交换第2维度和第3维度
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

# 生成图像
shap.image_plot(fin_shap_values, fin_explain, index_names)
