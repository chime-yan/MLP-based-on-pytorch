import torch
from sklearn import datasets, model_selection
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import FNN


# 手写数字数据集（测试softmax）
X = datasets.load_digits().data
y = datasets.load_digits().target
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

# 定义神经网络的结构
"""
    基本参数：
    - input_size: 输入层的特征数量
    - output_size: 输出维度
    
    结构超参数：
    - hidden_layers: 隐藏层数量
    - hidden_neurons: 每个隐藏层的神经元数量，列表格式
    
    结构基本参数：
    - activation_fn: 默认的隐藏层激活函数(默认为 ReLU)    
    - last_activation: 最后隐藏层到输出层的激活函数（可选）
    
    - dropout_way: 是否启用暂退法(默认为False)
    - dropout: 隐藏层丢弃概率(默认0.4)   # dropout 是结构超参数
    - flatten: 添加展平层（默认False）
    
    训练基本参数：
    - loss_fn: 损失函数（默认使用 CrossEntropyLoss）
    - optimizer: 优化器类型（默认为Adam）
    
    早停法参数：
    - model_path: 模型保存路径（默认'best_model.pth'）
    - patience: 验证集损失没有优化时，可容忍的无优化 epoch 的次数（默认为10）
    - delta: 定义验证集损失有效优化的减少幅度（默认为0.01）
    
    分类映射：
    - class_dict: 索引-类别映射（可选）
    
    训练超参数：
    - lr: 学习率（默认0.01）
    - weight_decay: L2正则化系数(默认0.0001)
    - lr_weaken_size: 经过多少 epoch 后，衰减lr（默认为10）
"""


input_size = X.shape[1]
output_size = len(np.unique(y))

hidden_layers = 3
hidden_neurons = [50, 25, 15]

activation_fn = None
last_layer_activation = None

dropout_way = True
dropout = 0.4
flatten = False

loss_fn = None

optimizer = None

model_path = 'best_model.pth'
patience = 6
delta = 0.01

class_dict = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}


lr = 0.01
weight_decay = 0.01


lr_weaken_size = 10

x_train = torch.tensor(x_train).to(torch.double)
y_train = torch.tensor(y_train).to(torch.long)

x_test = torch.tensor(x_test).to(torch.double)
y_test = torch.tensor(y_test).to(torch.long)


model = FNN.MLP(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers,
                hidden_neurons=hidden_neurons, activation_fn=activation_fn, last_activation=last_layer_activation,
                dropout_way=dropout_way, dropout=dropout, flatten=flatten, loss_fn=loss_fn, optimizer=optimizer,
                model_path=model_path, patience=patience, delta=delta, class_dict=class_dict)
model = model.double()


# model.forward(x_train)
# model.train_model(x_train, y_train, x_test, y_test, lr=lr, weight_decay=weight_decay, lr_weaken_size=lr_weaken_size)
# train_loss = model.train_loss
# print(f"Train Loss: {train_loss:.4f}")
# epoch_losses = model.epoch_losses

# 读取保存的模型
model_path = 'best_model_1.pth'
state_dict = torch.load(model_path, weights_only=True)

# 修改键名，将 "0.weight" 改为 "network.0.weight"
new_state_dict = {}
for key, value in state_dict.items():
    new_key = "network." + key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)


y_pred, test_loss = model.predict(x_test, y_test)

y_test = y_test.numpy()
y_pred = np.array(y_pred)

# print(f"Epoch Loss: {epoch_losses}")
print(f"Test Loss: {test_loss:.4f}")

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}")
