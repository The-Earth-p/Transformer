import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy

# embedding1 = nn.Embedding(10,3)
# input1 = torch.LongTensor([[1,6,1,8],[1,6,0,9]])
# print(embedding1(input1))
# print("-----------------------")
# embedding2 = nn.Embedding(10,3,padding_idx=0)
# input2 = torch.LongTensor([[0,6,1,8],[1,6,0,9]])
# print(embedding2(input2))
#  构建embedding类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        #vocab就是词汇表的总共大小
        super(Embeddings,self).__init__()
        #将一个二维的张量映射成最后一维为d_model长度的三维张量
        self.lut = nn.Embedding(vocab,d_model)
        self.d_model = d_model

    def forward(self,x):
        return self.lut(x) * math.sqrt(self.d_model)
#d_model代表一个词的表现维度，越大表现的越精确，也就是一个张量有多少列
d_model = 512
vocab = 1000
x = torch.LongTensor([[100,2,421,508],[491,998,1,221]])
emb = Embeddings(d_model,vocab)
embr = emb(x)
# print("embr: ",embr)
# #输入张量的形状是(batch_size, sequence_length)，
# #那么返回的张量形状将是(batch_size, sequence_length, d_model)
# print(embr.shape)

# m = nn.Dropout(p=0.2)
# input1 = torch.randn(4,5)
# output = m(input1)
# # print(output)
# x=torch.tensor([1,2,3,4])
# y=torch.unsqueeze(x,0)
# print(y,y.shape)
# z=torch.unsqueeze(x,1)
# print(z,z.shape)

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

dropout = 0.1
max_len = 60
x=embr
pe = PositionEncoding(d_model,dropout,max_len)
pe_result = pe(x)
# print(pe_result)
# print(pe_result.shape)

# plt.figure(figsize=(15,5))
# pe = PositionEncoding(20,0)
# #传入全0初始化x
# y = pe(torch.zeros(1,100,20))
# print(y)
# plt.plot(np.arange(100),y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d"%p for p in [4,5,6,7]])
# plt.show()