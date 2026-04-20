import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_add_pool
from torch.nn import Sequential, Linear, ReLU
import pandas as pd
df = pd.read_csv('./Dataset/data/hairpin/test/test_fold_5.csv')
# Step 1: 解析点括号形式，构建邻接矩阵和节点特征
def dot_bracket_to_graph(dot_bracket):
    n = len(dot_bracket)
    adjacency_matrix = np.zeros((n, n), dtype=int)  # 邻接矩阵
    node_features = []  # 节点特征（碱基类型和是否配对）

    # 碱基类型编码（A=0, U=1, C=2, G=3）
    base_to_index = {'A': 0, 'U': 1, 'C': 2, 'G': 3}

    # 随机生成一个RNA序列（假设）
    sequence = np.random.choice(['A', 'U', 'C', 'G'], size=n)

    stack = []
    for i, char in enumerate(dot_bracket):
        # 节点特征：碱基类型 + 是否配对（0未配对，1配对）
        base = sequence[i]
        base_feature = base_to_index[base]
        paired = 0  # 默认未配对
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                raise ValueError("点括号形式不匹配：缺少配对的 '('")
            j = stack.pop()
            adjacency_matrix[i][j] = 1
            adjacency_matrix[j][i] = 1
            paired = 1  # 配对
        node_features.append([base_feature, paired])

        # 添加磷酸二酯键（相邻碱基连接）
        if i < n - 1:
            adjacency_matrix[i][i + 1] = 1
            adjacency_matrix[i + 1][i] = 1

    # 检查是否有多余的 '('
    if stack:
        raise ValueError("点括号形式不匹配：缺少配对的 ')'")

    # 将邻接矩阵转换为边索引格式（PyTorch Geometric所需格式）
    edge_index = np.array(np.where(adjacency_matrix == 1))

    # 转换为PyTorch张量
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    return node_features, edge_index


# Step 2: 定义GIN模型（仅用于提取向量）
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GIN, self).__init__()
        # GIN卷积层
        self.conv1 = GINConv(
            Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        )
        self.conv2 = GINConv(
            Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        )

    def forward(self, x, edge_index, batch):
        # 图卷积
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # 全局池化，得到图级别的向量表示
        x = global_add_pool(x, batch)
        return x


# Step 3: 提取向量
def extract_vector(dot_bracket):
    # 转换为图数据
    node_features, edge_index = dot_bracket_to_graph(dot_bracket)
    data = Data(x=node_features, edge_index=edge_index)
    data.batch = torch.tensor([0] * len(node_features), dtype=torch.long)  # 批处理索引

    # 初始化GIN模型
    model = GIN(input_dim=2, hidden_dim=16)  # 输入维度2（碱基类型+是否配对），隐藏层维度16

    # 提取向量
    model.eval()
    with torch.no_grad():
        vector = model(data.x, data.edge_index, data.batch)

    return vector.numpy()

output_vectors = []
for i in df['RNAFolds']:
    dot_bracket = i
    vector = extract_vector(dot_bracket)
    output_vectors.append(vector)

# 将所有向量组合成一个大的列表
all_vectors = np.array(output_vectors)
all_vectors_flattened = all_vectors.reshape(all_vectors.shape[0], -1)

# 写入所有向量到 CSV 文件中
np.savetxt("./Dataset/data/hairpin/RNAfold/test/test_RNAfold_5.csv", all_vectors_flattened, delimiter=",")
