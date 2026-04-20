import os
import random
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from script.adjust_learning import *
from script.dataloader import *
from script.model import *
from script.model_V1 import *
from script.GIN import *
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as cData
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data
import math
import feat
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

sensitivity = []
specificity = []
accuracy = []
mcc_list = []

if __name__ == '__main__':
    # 设置随机种子
    seed_val = 1000
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    # GPU设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoches = 100
    
    # 数据集路径
    dataset_folder = 'Dataset'
    for i in range(0, 5):
        print(f'fold = {i + 1}:')
        for folder_name in os.listdir(dataset_folder):
            if not folder_name.startswith('.'):
                folder_path = os.path.join(dataset_folder, folder_name)
                
                # 文件路径
                train_file = f"Dataset/data/hairpin/train/train_newfold_{i + 1}.csv"
                test_file = f"Dataset/data/hairpin/test/test_newfold_{i + 1}.csv"
                train_feature_file = f"Dataset/data/hairpin/feature/train/trainoptimumDataset_newfold_{i + 1}.csv"
                test_feature_file = f"Dataset/data/hairpin/feature/test/testoptimumDataset_newfold_{i + 1}.csv"
                train_rnafold_file = f"Dataset/data/hairpin/RNAfold/train/train_newRNAfold_{i + 1}.csv"
                test_rnafold_file = f"Dataset/data/hairpin/RNAfold/test/test_newRNAfold_{i + 1}.csv"
                
                # 加载数据
                train_data = pd.read_csv(train_file)
                train_feature_data = pd.read_csv(train_feature_file)
                train_RNAfold_data = pd.read_csv(train_rnafold_file)
                train_sentences = train_data["rev"]
                train_labels = train_data["inblood"]
                
                test_data = pd.read_csv(test_file)
                test_feature_data = pd.read_csv(test_feature_file)
                test_RNAfold_data = pd.read_csv(test_rnafold_file)
                test_sentences = test_data["rev"]
                test_labels = test_data["inblood"]
                
                # 处理输入
                bert_path = './DNABERT-2-117M'
                train_inputs, train_labels, test_inputs, test_labels = input_token(
                    train_sentences, train_labels, test_sentences, test_labels, bert_path
                )
                
                # 转换为Tensor
                train_feature_embedding = torch.tensor(train_feature_data.to_numpy())
                train_rnafold_embedding = torch.tensor(train_RNAfold_data.to_numpy())
                test_feature_embedding = torch.tensor(test_feature_data.to_numpy())
                test_rnafold_embedding = torch.tensor(test_RNAfold_data.to_numpy())
                
                # 合并特征（确保离散特征为long类型）
                data1 = torch.cat(
                    [train_inputs.long(), train_feature_embedding, train_rnafold_embedding], 
                    dim=1
                ).float()
                data2 = torch.cat(
                    [test_inputs.long(), test_feature_embedding, test_rnafold_embedding],
                    dim=1
                ).float()
                
                # 计算模型参数
                vocab_size = int(torch.max(train_inputs)) + 1  # 词汇表大小
                embedding_dim = 64  # 嵌入维度
                other_feature_dim = train_feature_embedding.shape[1] + train_rnafold_embedding.shape[1]
                
                # 初始化模型
                bert_blend_cnn = Bert_Blend_CNN(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    input_channel_other=other_feature_dim
                ).to(device)
                
                # 优化器和损失函数
                optimizer = AdamW(bert_blend_cnn.parameters(), lr=1e-5, weight_decay=1e-2)
                loss_fn = nn.CrossEntropyLoss()
                
                # 数据加载器
                train_dataset = MyDataset(data1, train_labels)
                test_dataset = MyDataset(data2, test_labels)
                trainloader = cData.DataLoader(train_dataset, batch_size=32, shuffle=True)
                testloader = cData.DataLoader(test_dataset, batch_size=32, shuffle=True)
                
                print('%s:' % folder_name)
                for epoch in range(epoches):
                    print(f'Starting epoch {epoch + 1}')
                    print('Starting training')
                    # corrcet_number, total_number, real positive number, real and predict both are positive number
                    correct, total, pos_num, tp = 0, 0, 0, 0
                    bert_blend_cnn.train()  # 设置模型为训练模式
                    for i, batch in enumerate(trainloader):
                        optimizer.zero_grad()
                        # batch[0] is token embedding; batch[1] is real label
                        batch = tuple(p.to(device) for p in batch)
                        target = batch[1].long()
                        # 检查目标张量的值是否在合理范围内
                        if torch.min(target) < 0 or torch.max(target) >= 2000:
                            # 如果目标值不在合理范围内，进行适当处理，例如重新映射到有效的范围
                            target = torch.clamp(target, 0, 2000 - 1)  # 将目标值限制在有效范围内
                        # Data input to the model
                        pred = bert_blend_cnn(batch[0].float())
                        # Calculate loss function
                        loss = loss_fn(pred, target)
                        # Back Propagation
                        loss.backward()
                        # Warm-up and Learning Rate Decay
                        adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=epoches, lr_min=2e-6,
                                             lr_max=1.5e-5,
                                             warmup=True)
                        # Model weight update
                        optimizer.step()
                        # predicted label
                        _, predicted = torch.max(pred, 1)
                        total += batch[1].size(0)
                        # correct number
                        correct += (predicted == batch[1]).sum().item()
                        # positive number
                        pos_num += (batch[1] == 1).sum().item()
                        tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
                    neg_num = total - pos_num
                    tn = correct - tp
                    sn = (tp / pos_num) if pos_num != 0 else 1
                    sp = (tn / neg_num) if neg_num != 0 else 1
                    # Calculation accuracy
                    acc = (tp + tn) / (pos_num + neg_num) if (pos_num + neg_num) != 0 else 1
                    fn = pos_num - tp
                    fp = neg_num - tn
                    # Calculate Matthews correlation coefficient
                    mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) \
                    if (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) != 0 else 1
                    print('Sn = %.4f  Sp = %.4f  Acc = %.4f  Mcc= %.4f  ' % (sn,sp,acc,mcc))
                    print("train lr is ", optimizer.state_dict()["param_groups"][0]["lr"])
                    print('Starting testing')
                    # model valuing
                    correct, total, pos_num, tp = 0, 0, 0, 0
                    bert_blend_cnn.eval()  # 设置模型为评估模式                
                    with torch.no_grad():  # 不计算梯度
                        for i, batch in enumerate(testloader):
                            batch = tuple(p.to(device) for p in batch)
                            pred = bert_blend_cnn(batch[0].float())  # 使用训练的模型进行测试
                            _, predicted = torch.max(pred, 1)
                            total += batch[1].size(0)
                            correct += (predicted == batch[1]).sum().item()
                            pos_num += (batch[1] == 1).sum().item()
                            tp += ((batch[1] == 1) & (predicted == 1)).sum().item()
                    neg_num = total - pos_num
                    tn = correct - tp
                    sn = (tp / pos_num) if pos_num != 0 else 1
                    sp = (tn / neg_num) if neg_num != 0 else 1
                    acc =(tp + tn) / (pos_num + neg_num) if (pos_num + neg_num) != 0 else 1
                    fn = pos_num - tp
                    fp = neg_num - tn
                    mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) \
                        if (math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))) != 0 else 1
                    print('Sn = %.4f  Sp = %.4f  Acc = %.4f  Mcc= %.4f ' % (sn, sp, acc, mcc))
                    print('--------------------------------')
                    # saving model
                    if (epoch + 1) % 100 == 0:
                        print(f'epoch = {epoch + 1}. Saving trained model.')
                        save_path = os.path.join(folder_path, 'model.pth')
                        torch.save(bert_blend_cnn.state_dict(), save_path)