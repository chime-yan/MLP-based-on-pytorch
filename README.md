# MLP-based-on-pytorch  
Handwritten digit recognition and California house price prediction

## 1. 项目简介

本项目基于 PyTorch 实现了一个可高度定制的多层感知机（MLP）模型，并应用于两个经典任务：

- **手写数字识别**（分类任务） —— 使用 Scikit-learn 的 `load_digits` 数据集，验证模型在多类分类任务中的性能，输出准确率和混淆矩阵；
- **加利福尼亚房价预测**（回归任务） —— 使用 `fetch_california_housing` 数据集，评估模型在回归任务中的表现，输出预测值与 MSE 损失。

该项目支持如下功能：

- 多隐藏层结构；
- 激活函数和 Dropout 可灵活配置；
- 自定义损失函数与优化器；
- 支持早停（EarlyStopping）机制；
- 分类映射支持；
- 自动学习率衰减策略；
- 模型保存与加载机制；
- 通用分类/回归任务兼容。

---

## 2. 文件结构

```
.
├── FNN.py                              # 包含 MLP 网络结构、训练方法、预测方法、早停机制等核心代码
├── digits_classification.py            # 手写数字识别任务主脚本
├── housing_regression.py               # 加利福尼亚房价预测任务主脚本
├── best_model.pth                      # 分类任务保存的模型参数（如已生成）
├── california_housing_best_model.pth   # 回归任务保存的模型参数（如已生成）
├── california_housing_losses.png       # 回归任务训练损失曲线（如启用保存）
└── README.md                           # 项目说明文档（当前文件）
```

---

## 3. 超参数说明

在两个脚本中，可以灵活控制以下参数：

| 参数名               | 说明                                                     | 示例值或默认                                     |
|---------------------|----------------------------------------------------------|--------------------------------------------|
| `input_size`        | 输入特征数量                                             | 自动获取                                       |
| `output_size`       | 输出维度（分类：类别数，回归：1）                       | 分类: 10 / 回归: 1                             |
| `hidden_layers`     | 隐藏层层数                                               | 3                                          |
| `hidden_neurons`    | 每层隐藏层的神经元个数（列表）                           | [50, 25, 15]                               |
| `activation_fn`     | 隐藏层激活函数（默认为 ReLU）                            | `None`                                     |
| `last_activation`   | 最后隐藏层到输出层的激活函数                             | `None`                                     |
| `dropout_way`       | 是否启用 Dropout                                         | `True`                                     |
| `dropout`           | Dropout 概率                                             | 0.4                                        |
| `flatten`           | 是否启用展平层                                           | `False`                                    |
| `loss_fn`           | 自定义损失函数                                           | 分类: 默认 `CrossEntropyLoss`<br>回归: `MSELoss` |
| `optimizer`         | 自定义优化器                                             | `Adam`                                     |
| `patience`          | 早停机制允许无提升的最大 Epoch 数                        | 10                                         |
| `delta`             | 验证损失的最小改善幅度                                   | 0.01                                       |
| `lr`                | 学习率                                                   | 0.001                                      |
| `weight_decay`      | L2 正则化系数                                            | 0.0001                                     |
| `lr_weaken_size`    | 学习率衰减周期                                           | 分类: 10 / 回归: 13                            |

---

## 4. 权重初始化说明

在本项目中，为了提升网络的训练效率和稳定性，根据所选激活函数类型采用了不同的权重初始化策略：

- **He 正态分布初始化**（Kaiming normal distribution initialization）：适用于 ReLU 或其变种激活函数，能有效缓解梯度消失问题，提升训练深层网络时的稳定性。
- **Xavier 均匀分布初始化**（又称 Glorot 均匀分布初始化）：适用于 Tanh、Sigmoid 等对称型激活函数，可保持各层信号在前向传播过程中的方差一致，防止梯度爆炸或消失。

---

## 5. 学习率衰减机制说明

为进一步提升模型在训练后期的泛化能力，项目中使用了**StepLR 学习率调度器**来动态调整学习率。

该策略允许在训练过程中周期性地减小学习率，以便模型在接近最优解时进行更细致的参数调整。

---

## 6. 运行方法

**(1) 安装依赖**

```bash
pip install torch scikit-learn matplotlib numpy
```

**(2) 训练手写数字识别模型**

在 `digits_classification.py` 中取消以下行注释：
```python
# model.forward(x_train)
# model.train_model(x_train, y_train, x_test, y_test, lr=lr, weight_decay=weight_decay, lr_weaken_size=lr_weaken_size)
# train_loss = model.train_loss
# print(f"Train Loss: {train_loss:.4f}")
```

并将以下部分代码注释：
```python
# 读取保存的模型
model_path = 'best_model_1.pth'
state_dict = torch.load(model_path, weights_only=True)

# 修改键名，将 "0.weight" 改为 "network.0.weight"
new_state_dict = {}
for key, value in state_dict.items():
    new_key = "network." + key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
```

然后运行：
```bash
python 手写数字识别.py
```

**(3) 训练加利福尼亚房价回归模型**

在 `housing_regression.py` 中取消以下行注释：
```python
# model.forward(x_train)
# model.train_model(x_train, y_train, x_test, y_test, lr=lr, weight_decay=weight_decay, lr_weaken_size=lr_weaken_size)
# train_loss = model.train_loss
# print(f"Train Loss: {train_loss:.4f}")
```

并将以下部分代码注释：
```python
# 读取保存的模型
model_path = 'california_housing_best_model_1.pth'
state_dict = torch.load(model_path, weights_only=True)

# 修改键名，将 "0.weight" 改为 "network.0.weight"
new_state_dict = {}
for key, value in state_dict.items():
    new_key = "network." + key
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
```

然后运行：
```bash
python 预测加利福尼亚房价.py
```

---

## 7. 模型性能指标

### 手写数字识别（load_digits）

- 输出指标：
  - Accuracy 准确率
  - Confusion Matrix 混淆矩阵
  - CrossEntropyLoss 训练集和测试集损失

### 房价预测（California Housing）

- 输出指标：
  - MSE（均方误差）作为回归损失
  - 预测值与实际值打印对比
  - 可选训练曲线图 `california_housing_losses.png`

---

## 8. 注意事项

- 模型保存与加载时，键名统一添加前缀 `"network."`，确保模型结构兼容。
- 分类任务的 `output_size` 应等于类别数（如 10），回归任务为 1。
- 如果你使用 `weights_only=True` 加载 `.pth` 文件，确保你保存模型时使用 `torch.save(model.state_dict())`。

---

## 9. 联系方式

如有任何问题或建议，欢迎通过 issue 联系作者。该项目适用于课程设计、模型测试、基础任务集成等多种用途。