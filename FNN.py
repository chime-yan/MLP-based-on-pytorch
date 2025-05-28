import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_neurons, activation_fn=None, last_activation=None,
                 dropout_way=False, dropout=0.4, flatten=False, loss_fn=None, optimizer=None,
                 model_path='best_model.pth', patience=10, delta=0.01, class_dict=None):
        super(MLP, self).__init__()

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
                - patience: 验证集损失没有优化时，可容忍的无优化 epoch 的次数（默认为10） # patience 是训练超参数
                - delta: 定义验证集损失有效优化的减少幅度（默认为0.01）
                
                分类映射：
                - class_dict: 索引-类别映射（可选）
                
                训练超参数：
                - lr: 学习率（默认0.001）
                - weight_decay: L2正则化系数(默认0.0001)
                - lr_weaken_size: 经过多少 epoch 后，衰减lr（默认为10）
                
                - early_stopping: 早停法对象，不可更改
                
                训练损失：
                - train_loss: 训练损失
                - epoch_losses: 所有 epoch 的训练损失（列表形式）
        """

        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons

        self.activation_fn = activation_fn if activation_fn else nn.ReLU()
        self.last_activation = last_activation

        self.dropout_way = dropout_way
        self.dropout = dropout
        self.flatten = flatten

        self.loss_fn = loss_fn if loss_fn else nn.CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else 'Adam'

        self.model_path = model_path
        self.patience = patience
        self.delta = delta

        self.class_dict = class_dict

        self.early_stopping = EarlyStopping(patience=self.patience, delta=self.delta, path=self.model_path)

        self.lr = None
        self.weight_decay = None
        self.lr_weaken_size = None

        self.train_loss = None
        self.epoch_losses = None

        if self.output_size != 1:
            assert self.class_dict is not None

        # 确保隐藏层数量和神经元数量列表长度匹配
        assert self.hidden_layers == len(self.hidden_neurons)

        # 初始化存放各层的列表
        layers = []

        # 只有线性回归层（即线性回归模型）
        if self.hidden_layers == 0 and self.output_size == 1:
            if self.flatten:
                layers.append(nn.Flatten())

            layers.append(nn.Linear(self.input_size, self.output_size))

        # 无隐藏层
        elif self.hidden_layers == 0:
            if self.flatten:
                layers.append(nn.Flatten())

            layers.append(nn.Linear(self.input_size, self.output_size))

            if self.last_activation is not None:
                layers.append(self.last_activation)

            else:
                pass

        # 有隐藏层
        elif self.hidden_layers > 0:
            if self.flatten:
                layers.append(nn.Flatten())

            # 输入层到第一个隐藏层
            layers.append(nn.Linear(self.input_size, self.hidden_neurons[0]))
            layers.append(self.activation_fn)

            if self.hidden_layers > 1:
                # 第二层隐藏层及其之后的隐藏层
                for i in range(1, self.hidden_layers):
                    layers.append(nn.Linear(self.hidden_neurons[i - 1], self.hidden_neurons[i]))  # 前一层到当前隐藏层
                    layers.append(self.activation_fn)  # 隐藏层激活函数

                    if self.dropout_way:
                        layers.append(nn.Dropout(self.dropout))

            # 最后隐藏层到输出层
            layers.append(nn.Linear(self.hidden_neurons[-1], self.output_size))

            if self.last_activation is not None:
                layers.append(self.last_activation)

            else:
                pass

        else:
            pass

        # 使用 nn.Sequential 将所有层组合成一个网络
        self.network = nn.Sequential(*layers)

        # 初始化所有层权重和偏置
        self._init_params()

    # 初始化所有层的权重和偏置
    def _init_params(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):

                # He 初始化（适用relu型）
                if isinstance(self.activation_fn, nn.ReLU):
                    nn.init.kaiming_normal_(layer.weight)

                # Xavier 初始化（适用对称型）
                elif isinstance(self.activation_fn, nn.Tanh) or isinstance(self.activation_fn, nn.Sigmoid):
                    nn.init.xavier_uniform_(layer.weight)

                else:
                    pass

                if layer.bias is not None:  # 如果有偏置，初始化偏置
                    nn.init.zeros_(layer.bias)

            else:
                pass

    # 前向传播
    def forward(self, x):
        return self.network(x)

    # 计算损失
    def _compute_loss(self, model_outputs, y_true):
        return self.loss_fn(model_outputs, y_true)

    # 获取损失值
    def _get_loss(self, model_outputs, y_true):
        loss = self._compute_loss(model_outputs, y_true)
        return loss

    # 设置优化器
    def get_optimizer(self, optimizer, lr, weight_decay):
        if optimizer == 'Adam':
            return optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)

        elif optimizer == 'SGD':
            return optim.SGD(self.network.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

        else:
            raise NotImplementedError

    # 训练模型
    def train_model(self, train_x, train_y, val_x, val_y, batch_size=64, num_epochs=200, lr=0.001, weight_decay=0.0001,
                    lr_weaken_size=10, verbose_batch=False, verbose=False, val_verbose=False):

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_weaken_size = lr_weaken_size

        # 创建数据集和 DataLoader
        dataset = TensorDataset(train_x, train_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = self.get_optimizer(self.optimizer, self.lr, weight_decay=self.weight_decay)

        # 设置学习率衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_weaken_size, gamma=0.1)

        epoch_losses = []
        for epoch in range(num_epochs):
            self.network.train()

            total_loss = 0
            total_samples = 0

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                outputs = self.forward(inputs)
                if self.output_size == 1:
                    outputs = outputs.squeeze(-1)

                train_loss = self._get_loss(outputs, labels)

                # 累积损失加权
                batch_count = inputs.size(0)
                total_loss += train_loss.item() * batch_count
                total_samples += batch_count

                # 反向传播，更新参数
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                if verbose_batch:
                    # 打印每一 batch 的信息
                    if (batch_idx + 1) % batch_size == 0:  # 每个 batch 打印一次
                        print(
                            f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {train_loss.item():.4f}")

            # 衰减学习率
            scheduler.step()

            # 计算一个 epoch 总的平均损失
            epoch_loss = total_loss / total_samples
            self.train_loss = epoch_loss

            if verbose:
                # 打印每一 epoch 的信息
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss.item():.4f}")

            epoch_losses.append(epoch_loss)
            self.epoch_losses = epoch_losses

            # 验证过程
            self.network.eval()
            with torch.no_grad():
                val_outputs = self.network(val_x)
                if self.output_size == 1:
                    val_outputs = val_outputs.squeeze(-1)

                val_loss = self._get_loss(val_outputs, val_y)
                if val_verbose:
                    print(f"Validation Loss: {val_loss.item():.4f}")

            # 早停检查
            self.early_stopping(val_loss.item(), self.network)

            # 如果早停触发，停止训练
            if self.early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # 预测及损失计算
    def predict(self, test_x, test_y=None):
        self.network.eval()
        with torch.no_grad():
            if self.last_activation is None and self.output_size == 1:
                pred_y = self.forward(test_x).squeeze(-1)

                if test_y is not None:
                    pred_loss = self._get_loss(pred_y, test_y)

            elif isinstance(self.loss_fn, nn.CrossEntropyLoss) and self.class_dict is not None:
                model_outputs = self.forward(test_x)

                if test_y is not None:
                    pred_loss = self._get_loss(model_outputs, test_y)

                vector_y = F.softmax(model_outputs, dim=1)
                _, pred_y = torch.max(vector_y, 1)

                predicted_classes = [self.class_dict[idx.item()] for idx in pred_y]
                pred_y = predicted_classes

            elif isinstance(self.loss_fn, nn.BCEWithLogitsLoss) and self.class_dict is not None:
                model_outputs = self.forward(test_x)

                if test_y is not None:
                    pred_loss = self._get_loss(model_outputs, test_y)

                vector_y = F.sigmoid(model_outputs)
                _, pred_y = torch.max(vector_y, 1)

                predicted_classes = [self.class_dict[idx.item()] for idx in pred_y]
                pred_y = predicted_classes

            else:
                pass

        if test_y is not None:
            return pred_y, pred_loss
        else:
            return pred_y

    # 打印模型所有的参数
    def print_parameters(self, print_grad=False):
        for name, param in self.network.parameters():
            print(f"Parameter Name: {name}, Shape: {param.shape}")
            print(param.data)
            if print_grad:
                print(param.grad)


class EarlyStopping:
    def __init__(self, patience=10, delta=0.01, path='best_model.pth'):
        """
        参数：
        patience: 容忍多少个epoch验证集性能没有提升
        delta: 用于定义“性能提升”的阈值
        """

        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True
