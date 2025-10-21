from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
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
        """简单的文本清洗"""
        # 移除非字母数字字符，但保留基本标点和空格
        text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
        # 规范化空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def __len__(self):
        """返回数据集大小"""
        return len(self.src_sentences)

    def __getitem__(self, idx):
        """获取一个训练样本"""
        # 获取原始句子
        src_text = self.src_sentences[idx]
        tgt_text = self.tgt_sentences[idx]

        # 使用tokenizer编码句子
        # 源语言（英语）编码
        src_encoded = self.tokenizer(
            src_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 目标语言（德语）编码，添加开始和结束标记
        tgt_encoded = self.tokenizer(
            tgt_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # 提取输入id和注意力掩码
        src_input_ids = src_encoded['input_ids'].squeeze()
        src_attention_mask = src_encoded['attention_mask'].squeeze()
        tgt_input_ids = tgt_encoded['input_ids'].squeeze()
        tgt_attention_mask = tgt_encoded['attention_mask'].squeeze()

        return {
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            'tgt_input_ids': tgt_input_ids,
            'tgt_attention_mask': tgt_attention_mask
        }


# 创建掩码的辅助函数
def create_masks(src_input_ids, tgt_input_ids, pad_token_id):
    """创建源语言和目标语言的掩码

    Args:
        src_input_ids: 源语言输入ID张量 [batch_size, seq_len]
        tgt_input_ids: 目标语言输入ID张量 [batch_size, seq_len]
        pad_token_id: 填充标记的ID

    Returns:
        源语言掩码和目标语言掩码
    """
    # 源语言掩码，用于屏蔽填充标记
    src_mask = (src_input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)

    # 目标语言掩码，包括填充掩码和后续掩码
    tgt_pad_mask = (tgt_input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    tgt_seq_len = tgt_input_ids.size(1)
    tgt_subsequent_mask = torch.tril(torch.ones((tgt_seq_len, tgt_seq_len))).bool()
    tgt_mask = tgt_pad_mask & tgt_subsequent_mask

    return src_mask, tgt_mask


# 创建数据加载器
def create_translation_dataloader(src_file, tgt_file, model_name='google-bert/bert-base-multilingual-cased',
                                  batch_size=32, max_length=100, shuffle=True, num_workers=0):
    """创建翻译数据加载器

    Args:
        src_file: 源语言文件路径
        tgt_file: 目标语言文件路径
        model_name: Hugging Face模型名称，用于获取对应的tokenizer
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        num_workers: 加载数据的进程数

    Returns:
        数据加载器和tokenizer实例
    """
    # 加载预训练的多语言tokenizer
    print(f"加载预训练的tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 创建数据集
    dataset = TranslationDataset(src_file, tgt_file, tokenizer, max_length)

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            'src_input_ids': torch.stack([item['src_input_ids'] for item in batch]),
            'src_attention_mask': torch.stack([item['src_attention_mask'] for item in batch]),
            'tgt_input_ids': torch.stack([item['tgt_input_ids'] for item in batch]),
            'tgt_attention_mask': torch.stack([item['tgt_attention_mask'] for item in batch])
        }
    )

    return dataloader, tokenizer


# 使用示例
if __name__ == "__main__":
    # 定义文件路径
    data_dir = "D:\\PostGrdu\\WorkSpace\\pythonProject\\Transformer"
    src_file = os.path.join(data_dir, "train.en")  # 英语文件
    tgt_file = os.path.join(data_dir, "train.de")  # 德语文件

    # 创建数据加载器
    train_loader, tokenizer = create_translation_dataloader(
        src_file, tgt_file,
        model_name='google-bert/bert-base-multilingual-cased',
        batch_size=4,
        max_length=50,
        shuffle=True,
        num_workers=0
    )

    # 测试数据加载器
    for batch in train_loader:
        print(f"源语言输入ID形状: {batch['src_input_ids'].shape}")
        print(f"源语言注意力掩码形状: {batch['src_attention_mask'].shape}")
        print(f"目标语言输入ID形状: {batch['tgt_input_ids'].shape}")
        print(f"目标语言注意力掩码形状: {batch['tgt_attention_mask'].shape}")

        # 解码一个样本进行检查
        sample_idx = 0
        src_ids = batch['src_input_ids'][sample_idx].tolist()
        tgt_ids = batch['tgt_input_ids'][sample_idx].tolist()

        # 解码
        src_text = tokenizer.decode(src_ids, skip_special_tokens=True)
        tgt_text = tokenizer.decode(tgt_ids, skip_special_tokens=True)

        print(f"源语言示例: {src_text}")
        print(f"目标语言示例: {tgt_text}")

        # 只打印一个批次
        break