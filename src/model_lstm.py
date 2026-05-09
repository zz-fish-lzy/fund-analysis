"""
LSTM模型模块
用于基金净值预测
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FundDataset(Dataset):
    """基金数据集"""
    def __init__(self, features, targets, seq_length=20):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return x, y


class FundLSTM(nn.Module):
    """LSTM基金预测模型"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(FundLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = lstm_out[:, -1, :]

        # 全连接层
        out = self.fc(out)

        return out


class LSTMPredictor:
    """LSTM预测器封装类"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, seq_length=20, lr=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_length = seq_length

        # 创建模型
        self.model = FundLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        ).to(self.device)

        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=100, batch_size=32, verbose=True):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
            verbose: 是否显示训练过程
        """
        # 创建数据集
        train_dataset = FundDataset(X_train, y_train, self.seq_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if X_val is not None:
            val_dataset = FundDataset(X_val, y_val, self.seq_length)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 训练历史
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)

            # 验证
            if X_val is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}")

        return history

    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                total_loss += loss.item()

        return total_loss / len(data_loader)

    def predict(self, X):
        """
        预测

        Args:
            X: 输入特征

        Returns:
            numpy.ndarray: 预测结果
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            prediction = self.model(X_tensor)
            return prediction.cpu().numpy().flatten()

    def predict_batch(self, X_batch):
        """
        批量预测

        Args:
            X_batch: 批量输入特征

        Returns:
            numpy.ndarray: 预测结果
        """
        self.model.eval()

        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_batch).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().flatten()

    def save(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已加载: {path}")


def calculate_accuracy(predictions, targets, threshold=0.0):
    """
    计算方向预测准确率

    Args:
        predictions: 预测值
        targets: 实际值
        threshold: 阈值

    Returns:
        float: 准确率
    """
    pred_direction = (predictions > threshold).astype(int)
    target_direction = (targets > threshold).astype(int)

    accuracy = np.mean(pred_direction == target_direction)
    return accuracy


if __name__ == '__main__':
    # 测试LSTM模型
    print("测试LSTM模型...")

    # 生成测试数据
    input_size = 19  # 特征数量
    seq_length = 20
    batch_size = 32

    # 创建模型
    predictor = LSTMPredictor(input_size=input_size, seq_length=seq_length)

    # 打印模型结构
    print(f"\n模型结构:")
    print(predictor.model)
    print(f"\n使用设备: {predictor.device}")

    # 测试前向传播
    test_input = torch.randn(batch_size, seq_length, input_size).to(predictor.device)
    output = predictor.model(test_input)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
