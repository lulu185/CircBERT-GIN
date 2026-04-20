import torch
import torch.nn as nn
import torch.nn.functional as F
from .CBAM import *

# 构建CNN模块和CBAM
class CNNNET2(nn.Module):
    def __init__(self, input_channel, embedding_dim=159, num_classes=2):
        super(CNNNET2, self).__init__()
        
        # 确保 input_channel 是整数
        if isinstance(input_channel, torch.Tensor):
            if input_channel.numel() != 1:
                input_channel = input_channel.mean().item()  # 取平均并转换为浮点数
            else:
                input_channel = input_channel.item()  # 转换为浮点数
        
        # 将 input_channel 转换为整数
        input_channel = int(round(input_channel))  # 四舍五入并转换为整数
        
        # 检查 input_channel 是否为正整数
        if not isinstance(input_channel, int) or input_channel <= 0:
            raise ValueError(f"input_channel 必须是正整数，当前值为: {input_channel}")
        
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channel, out_channels=60, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(60),
            nn.GELU()
            
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


# 构建Bert_Blend_CNN模型，仅包含CNN部分
class Bert_Blend_CNN2(nn.Module):
    def __init__(self, input_channel):
        super(Bert_Blend_CNN2, self).__init__()
        
        # 确保 input_channel 是整数
        if isinstance(input_channel, torch.Tensor):
            if input_channel.numel() != 1:
                input_channel = input_channel.mean().item()  # 取平均并转换为浮点数
            else:
                input_channel = input_channel.item()  # 转换为浮点数
        
        # 将 input_channel 转换为整数
        input_channel = int(round(input_channel))  # 四舍五入并转换为整数
        
        # 检查 input_channel 是否为正整数
        if not isinstance(input_channel, int) or input_channel <= 0:
            raise ValueError(f"input_channel 必须是正整数，当前值为: {input_channel}")
        
        self.model = CNNNET2(input_channel)

    def forward(self, x):
        x = self.model(x)
        return x