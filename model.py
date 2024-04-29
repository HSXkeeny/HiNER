# 定义robert上的模型结构
class ConvolutionLayer(nn.Module):

    def __init__(self, input_size, channels, dilation, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.base = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(input_size, channels, kernel_size=1),
            nn.GELU(),
        )

        self.convs = nn.ModuleList(
            [nn.Conv2d(channels, channels, kernel_size=3, groups=channels, dilation=d, padding=d) for d in dilation])

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.base(x)

        outputs = []
        for conv in self.convs:
            x = conv(x)
            x = F.gelu(x)
            outputs.append(x)
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 3, 1).contiguous()
        return outputs

# Triformer模型结构
class Triformer(nn.Module):
   
  def __init__(self, d_model, d_char, d_lex, d_syn, n_heads, n_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # 将输入嵌入映射到模型维度
        self.char_embed_proj = nn.Linear(d_char, d_model)
        self.lex_embed_proj = nn.Linear(d_lex, d_model)
        self.syn_embed_proj = nn.Linear(d_syn, d_model)
        
        # 定义相对位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 定义编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 多头注意力层
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn_char_lex = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn_char_syn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn_lex_syn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, char_embeddings, lex_embeddings, syn_embeddings):
        # 映射输入嵌入到模型维度
        char_embeddings = self.char_embed_proj(char_embeddings)
        lex_embeddings = self.lex_embed_proj(lex_embeddings)
        syn_embeddings = self.syn_embed_proj(syn_embeddings)
        
        # 添加位置编码
        char_embeddings = self.pos_encoder(char_embeddings)
        lex_embeddings = self.pos_encoder(lex_embeddings)
        syn_embeddings = self.pos_encoder(syn_embeddings)
        
        # 自注意力
        char_output = self.transformer_encoder(char_embeddings)
        lex_output = self.transformer_encoder(lex_embeddings)
        syn_output = self.transformer_encoder(syn_embeddings)
        
        # 交叉注意力
        char_lex_output, _ = self.cross_attn_char_lex(char_output, lex_output, lex_output)
        char_syn_output, _ = self.cross_attn_char_syn(char_output, syn_output, syn_output)
        lex_syn_output, _ = self.cross_attn_lex_syn(lex_output, syn_output, syn_output)
        
        # 融合嵌入
        fused_output = char_lex_output + char_syn_output + lex_syn_output
        fused_output = self.ffn(fused_output)
        
        return fused_output

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 放射变化
class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s

# MLP
class MLP(nn.Module):

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x

# 注意力融合模块
class AttentionFusion(nn.Module):
    def __init__(self, input_dim, window_size, beta=0.5):
        super(AttentionFusion, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.beta = beta
        self.local_attention = LocalAttention(input_dim, window_size)
        self.global_attention = GlobalAttention(input_dim)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        local_attn = self.local_attention(inputs)
        global_attn = self.global_attention(inputs)
        fused_attn = self.beta * local_attn + (1 - self.beta) * global_attn
        outputs = torch.bmm(fused_attn.view(batch_size, 1, seq_len * seq_len), inputs.view(batch_size, seq_len * seq_len, self.input_dim))
        return outputs

class LocalAttention(nn.Module):
    def __init__(self, input_dim, window_size):
        super(LocalAttention, self).__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        queries = self.query(inputs).view(batch_size, seq_len, self.input_dim)
        keys = self.key(inputs).view(batch_size, seq_len, self.input_dim)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        padded_mask = torch.ones(batch_size, seq_len, seq_len)
        padding_window = self.window_size // 2
        padded_mask[:, :, :padding_window] = 0
        padded_mask[:, :, -padding_window:] = 0
        for i in range(padding_window, seq_len - padding_window):
            padded_mask[:, i, i - padding_window:i + padding_window + 1] = 1
        attention_scores = attention_scores.masked_fill(padded_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights

class GlobalAttention(nn.Module):
    def __init__(self, input_dim):
        super(GlobalAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        queries = self.query(inputs).view(batch_size, seq_len, self.input_dim)
        keys = self.key(inputs).view(batch_size, seq_len, self.input_dim)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights

# 协同预测器
class CoPredictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, channels, ffnn_hid_size, dropout=0):
        super().__init__()
        self.mlp1 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp2 = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.mlp_rel = MLP(channels, ffnn_hid_size, dropout=dropout)
        self.linear = nn.Linear(ffnn_hid_size, cls_num)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y, z):
        h = self.dropout(self.mlp1(x))
        t = self.dropout(self.mlp2(y))
        o1 = self.biaffine(h, t)

        z = self.dropout(self.mlp_rel(z))
        o2 = self.linear(z)
        return o1 + o2
