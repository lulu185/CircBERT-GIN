import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import *

class CNNNET(nn.Module):
    def __init__(self, input_channel, num_classes=2):
        super(CNNNET, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU(),
        )
        
        # 第二层卷积（三个分支）
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU()
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU()
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv1d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=30, out_channels=60, kernel_size=3, stride=1, dilation=4, padding=4, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU()
        )
        
        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=180, out_channels=180, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(180)
        )
        
        # 修正linear1的输入维度为180
        self.linear1 = nn.Linear(180, 180)
        
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(180, num_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        text1 = self.conv2_1(x)
        text2 = self.conv2_2(x)
        text3 = self.conv2_3(x)
        x = torch.cat([text1, text2, text3], dim=1)
        x = self.conv3(x)
        x = x.view(x.shape[0], -1)
        
        x = self.linear1(x)
        x = self.drop(x)
        x = self.linear2(x)
        
        return F.softmax(x, dim=1)

class Bert_Blend_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_channel_other):
        super(Bert_Blend_CNN, self).__init__()
        # 嵌入层处理离散特征（假设前23列是token IDs）
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                    embedding_dim=embedding_dim,
                                    padding_idx=0)  # 假设0是padding索引
        
        # 计算总输入维度：嵌入后的特征 + 其他特征
        self.total_input_dim = embedding_dim + input_channel_other
        
        # CNN部分（输入维度调整为total_input_dim）
        self.cnn = CNNNET(input_channel=self.total_input_dim)

    def forward(self, x):
        # 分离离散特征（前23列）和其他特征
        discrete_features = x[:, :23].long()  # 强制转换为long类型
        other_features = x[:, 23:]
        
        # 嵌入层处理离散特征 [batch, 23] -> [batch, 23, embed_dim]
        embedded = self.embedding(discrete_features)
        
        # 沿序列维度池化（取均值）
        embedded_pooled = embedded.mean(dim=1)  # [batch, embed_dim]
        
        # 拼接其他特征
        combined = torch.cat([embedded_pooled, other_features], dim=1)
        
        # 输入到CNN
        output = self.cnn(combined)
        return output