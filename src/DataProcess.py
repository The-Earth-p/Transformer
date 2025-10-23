from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import re


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length=100):
        """初始化翻译数据集

        Args:
            src_file: 源语言文件路径
            tgt_file: 目标语言文件路径
            tokenizer: Hugging Face AutoTokenizer实例
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 读取源语言和目标语言文件
        self.src_sentences = self._read_file(src_file)
        self.tgt_sentences = self._read_file(tgt_file)

        # 确保源语言和目标语言句子数量一致
        assert len(self.src_sentences) == len(self.tgt_sentences), \
            f"源语言句子数量({len(self.src_sentences)})与目标语言句子数量({len(self.tgt_sentences)})不一致"

    def _read_file(self, file_path):
        """读取文本文件"""
        sentences = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # 简单的文本清洗
                        line = self._clean_text(line)
                        sentences.append(line)
        except Exception as e:
            print(f"读取文件{file_path}时出错: {e}")
        return sentences

    def _clean_text(self, text):
        # 简单的文本清洗
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]
        
        # 文本清洗
        src_text = self._clean_text(src_text)
        tgt_text = self._clean_text(tgt_text)
        
        # 分词
        src_encodings = self.tokenizer(
            src_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        tgt_encodings = self.tokenizer(
            tgt_text, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        # 准备目标输入和标签（标签需要偏移一位）
        tgt_input = tgt_encodings['input_ids'][:, :-1]
        tgt_label = tgt_encodings['input_ids'][:, 1:]
        
        return {
            'src_input_ids': src_encodings['input_ids'].squeeze(),
            'src_attention_mask': src_encodings['attention_mask'].squeeze(),
            'tgt_input_ids': tgt_input.squeeze(),
            'tgt_attention_mask': tgt_encodings['attention_mask'][:, :-1].squeeze(),
            'labels': tgt_label.squeeze()
        }

# 创建分词器
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        'google-bert/bert-base-multilingual-cased',
        do_lower_case=False
    )
    return tokenizer

# 创建数据集并划分训练集、验证集、测试集
def create_translation_datasets(src_file, tgt_file, tokenizer, val_ratio=0.1, test_ratio=0.1, max_length=128):
    # 创建完整数据集
    full_dataset = TranslationDataset(src_file, tgt_file, tokenizer, max_length)
    
    # 计算划分大小
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    test_size = int(total_size * test_ratio)
    train_size = total_size - val_size - test_size
    
    # 随机划分数据集
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 设置随机种子确保结果可复现
    )
    
    return train_dataset, val_dataset, test_dataset

# 创建数据加载器
def create_translation_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda x: x  # 使用默认的collate_fn
    )

# 为Transformer模型创建掩码
def create_masks(src, tgt, pad_token_id):
    device = src.device  # 获取设备信息

    # 源语言输入的掩码
    src_mask = (src != pad_token_id).unsqueeze(-2).to(device)

    # 目标语言的掩码（包括填充掩码和未来信息掩码）
    tgt_mask = (tgt != pad_token_id).unsqueeze(-2).to(device, dtype=torch.bool)
    seq_len = tgt.size(1)
    nopeak_mask = torch.tril(torch.ones((1, seq_len, seq_len), device=device, dtype=torch.bool))

    # 现在执行按位与操作
    tgt_mask = tgt_mask & nopeak_mask

    return src_mask, tgt_mask