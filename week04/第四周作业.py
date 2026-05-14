import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 缩放点积注意力（Transformer核心计算单元）
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        q: 查询 [batch_size, n_heads, seq_len, d_k]
        k: 键   [batch_size, n_heads, seq_len, d_k]
        v: 值   [batch_size, n_heads, seq_len, d_v]
        mask: 掩码 [batch_size, 1, seq_len, seq_len]
        """
        d_k = q.size(-1)
        # 计算注意力分数：Q*K^T / √d_k
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        
        # 掩码：将需要屏蔽的位置置为 -inf，softmax后为0
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重 + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

# 2. 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model  # 模型总维度
        self.n_heads = n_heads  # 注意力头数
        assert d_model % n_heads == 0, "模型维度必须能被头数整除"
        
        self.d_k = d_model // n_heads  # 每个头的维度
        # Q/K/V 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 线性投影 + 拆分为多头
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)
        
        # 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终投影 + dropout
        output = self.dropout(self.w_o(attn_output))
        return output, attn_weights

# 3. 前馈神经网络（FFN）
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # 论文标准结构：Linear → ReLU → Dropout → Linear
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# 4. 完整 Transformer 编码器层（核心！）
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # 多头注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈网络
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        # 层归一化（论文：Pre-LN）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: 输入 [batch_size, seq_len, d_model]
        mask: 注意力掩码
        """
        # 残差连接 + 层归一化（标准Pre-LN结构）
        # 第一步：自注意力
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 第二步：前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x

# ===================== 测试代码 =====================
if __name__ == "__main__":
    # 超参数（和论文一致）
    d_model = 512    # 模型维度
    n_heads = 8      # 注意力头数
    d_ff = 2048      # 前馈网络中间维度
    batch_size = 2   # 批次大小
    seq_len = 10     # 序列长度

    # 初始化Transformer编码器层
    encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
    
    # 构造随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = encoder_layer(x)
    
    # 打印结果
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("✅ Transformer编码器层实现成功！")
