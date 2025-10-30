import copy
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        #vocab就是词汇表的总共大小
        super(Embeddings,self).__init__()
        #将一个二维的张量映射成最后一维为d_model长度的三维张量
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model

    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionEncoding,self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len,d_model)#最终的位置编码
        position = torch.arange(0,max_len).unsqueeze(1)#position最终能有max_len行，是绝对位置
        #变换矩阵，进行跳跃式的初始化，是为了确保矩阵中各值能够在不同位置间产生更大的差异
        #所以这个矩阵中的各个值之间已经足够大的差异，最终得到的是频率成分
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0) / d_model))
        #对前面的矩阵进行奇数偶数分别的赋值，每个数学表达式生成的都是(max_len,d_model/2)的矩阵
        pe[:,0::2] = torch.sin(position*div_term)#有自动广播功能，保证矩阵能够相乘
        pe[:,1::2] = torch.cos(position*div_term)
        pe=pe.unsqueeze(0)#扩充为3维向量
        self.register_buffer('pe',pe)#保存模型后重新加载时可以直接加载这个参数

    def forward(self,x):
        #pe编码太长，也就是第二个维度，及它本来是一个句子的最长的长度，但是
        #一般的句子没有那么长，也就是每个向量没有那么多的行数，所以把这个行数
        #和输入的词嵌入向量相匹配，也就是每个向量的行数相匹配
        x = x+self.pe[:,:x.size(1)].clone().detach().requires_grad_(False)
        return self.dropout(x)

def attention(query,key,value,mask=None,dropout=None):
    #先提取query的最后一个维度，也就是词嵌入的维度，也就是一个词的张量有多少列
    d_k = query.size(-1)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask=mask, value=torch.tensor(-1e9))
    #p_attn表示注意力的权重,表示了在对应位置上，key对query的重要程度
    #p_attn第一行就是[1,0,0,0]表示第一个query和第一个key也就是关键词的管理度最高
    #用之前的比喻，一段话中的第一个query和第一个关键词最相关
    #如果score每一行都是相等的话，输出每行每个数据就是1/每行数据个数
    p_attn = F.softmax(scores,dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn,value), p_attn

#克隆函数，需要多个结构相同的线性层，所以一同初始化到网络层列表中
def clone(module,N):
    #module是要克隆到的网络层列表
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self,head,embedding_dim,dropout = 0.1):
        super(MultiHeadedAttention,self).__init__()
        assert embedding_dim % head == 0
        #d_k表示每个注意力头可以分到的词嵌入维度数量，也就是张量的最后一维-列数
        self.d_k = embedding_dim//head
        self.head = head
        self.embedding_dim = embedding_dim
        self.linears = clone(nn.Linear(embedding_dim,embedding_dim),4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            #扩充1方向上的维度，代表多头中的第n个头
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        #zip能够将网络层和输入数据连接在一起，for循环循环3次，分别得到QKV
        #让QKV经过线性层后再通过view函数进行分割，变成一个四维张量，因为要考虑多个头
        #克隆生成的线性层是随机初始化的，所以QKV最后得到的值会不同，增强模型表现能力
        query,key,value = \
        [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
         for model,x in zip(self.linears,(query,key,value))]
        x,self.attn = attention(query,key,value,mask = mask ,dropout=self.dropout)
        #恢复维度
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        return self.linears[-1](x)

#防止拟合不够
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()

        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,x):
        #x是上一层的输出
        return self.w2(self.dropout(F.relu(self.w1(x))))

#让数值变得合理
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        #eps防止标准差为0时又被当作除数
        super(LayerNorm,self).__init__()
        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self,x):
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim = True)
        return self.a2*(x-mean)/(std+self.eps)+self.b2

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        #size是词嵌入维度
        super().__init__()
        self.norm = LayerNorm(size)#规范化层已经包含在子层连接结构中了
        self.dropout = nn.Dropout(p=dropout)
        self.size = size

    def forward(self,x,sublayer):
        #sublayer代表这里所说的“子层”到底是一个什么层
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self,size,self_attn,feed_forward,dropout):
        #self_attn代表传入的多头自注意力子层的实例化对象
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size
        #编码器中有两个子层连接结构
        self.sublayer = clone(SublayerConnection(size,dropout),2)

    def forward(self,x,mask):
        #x代表传入张量,即已经经过词嵌入编码的张量
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)
class Encoder(nn.Module):
    def __init__(self,layer,N):
        #layer代表一层编码器层，N表示有几层
        super().__init__()
        self.layers = clone(layer,N)
        self.norm = LayerNorm(layer.size)#在最后面实现一个规范化

    def forward(self,x,mask):
        for layer in self.layers:
            x= layer(x,mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn#常规多头注意力对象
        self.feed_forward = feed_forward
        self.dropout = dropout

        self.sublayer = clone(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,source_mask,target_mask):
        #memory代表编码器的语义存储张量，也就是encoder层的最终输出
        #source_mask代表源数据掩码张量，目的是减少对结果信息无用的数据的影响
        #target_mask代表目的数据张量，遮掩未来信息，防止某一个词的未来信息影响模型训练
        m = memory
        x = self.sublayer[0](x, lambda x:self.self_attn(x,x,x,target_mask))
        x = self.sublayer[1](x,lambda x:self.src_attn(x,m,m,source_mask))
        return self.sublayer[2](x,self.feed_forward)


class Decoder(nn.Module):
    def __init__(self,layer,N):
        super().__init__()
        self.layers = clone(layer,N)
        self.norm = LayerNorm(layer.size)
    def forward(self,x,memory,source_mask,target_mask):
        for layer in self.layers:
            x = layer(x,memory,source_mask,target_mask)
        return self.norm(x)

class Generator(nn.Module):
    def __init__(self,d_model,vocab_size):
        super().__init__()
        #改变最终维度，对应到整个词汇表
        self.project = nn.Linear(d_model,vocab_size)

    def forward(self,x):
        #softmax层就是把数字放缩到0-1的概率中
        return F.log_softmax(self.project(x),dim=-1)

class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,source_embed,target_embed,generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self,source,target,source_mask,target_mask):
        return self.decode(self.encode(source,source_mask),source_mask,target,target_mask)
    def encode(self,source,source_mask):
        return self.encoder(self.src_embed(source),source_mask)
    def decode(self,memory,source_mask,target,target_mask):
        return self.decoder(self.tgt_embed(target),memory,source_mask,target_mask)

def make_model(
    source_vocab,
    target_vocab,
    N=6,
    d_model=512,
    d_ff=2048,
    head=8,
    dropout=0.1,
    use_pos_encoding=True  # ✅ 新增
):
    c = copy.deepcopy

    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)

    # ✅ 根据实验配置选择是否使用位置编码
    if use_pos_encoding:
        position = PositionEncoding(d_model, dropout)
    else:
        position = nn.Identity()  # 直接跳过位置编码

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab)
    )

    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return model