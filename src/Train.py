import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import random
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from TransformerModel import make_model
from DataProcess import get_tokenizer, create_translation_datasets, create_translation_dataloader, create_masks


# ===============================
# 模型配置
# ===============================
CONFIG = {
    'src_vocab_size': 10000,
    'tgt_vocab_size': 10000,
    'd_model': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'dim_feedforward': 2048,
    'dropout': 0.1
}


# ===============================
# 检查设备一致性
# ===============================
def check_device_consistency(model, src_input_ids, tgt_input_ids, src_mask, tgt_mask, labels):
    print("\n=== 设备一致性检查 ===")
    print(f"模型参数设备: {next(model.parameters()).device}")
    print(f"源输入设备: {src_input_ids.device}")
    print(f"目标输入设备: {tgt_input_ids.device}")
    print(f"源掩码设备: {src_mask.device}")
    print(f"目标掩码设备: {tgt_mask.device}")
    print(f"标签设备: {labels.device}")
    devices = [src_input_ids.device, tgt_input_ids.device, src_mask.device, tgt_mask.device, labels.device]
    if all(d == devices[0] for d in devices):
        print("✅ 所有张量在相同设备上\n")
    else:
        print("❌ 张量设备不一致！！\n")


# ===============================
# 验证阶段
# ===============================
def evaluate(model, dataloader, criterion, pad_token_id, device):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src_input_ids = torch.stack([item['src_input_ids'] for item in batch]).to(device)
            tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch]).to(device)
            labels = torch.stack([item['labels'] for item in batch]).to(device)

            # ✅ 保证 mask 也在同一设备上
            src_mask, tgt_mask = create_masks(src_input_ids, tgt_input_ids, pad_token_id)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            decoder_out = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)
            logits = model.generator(decoder_out)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            non_pad_tokens = (labels != pad_token_id).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0

def calculate_bleu(model, dataloader, tokenizer, device, max_len=64):
    model.eval()
    smooth = SmoothingFunction().method1
    total_bleu = 0
    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating BLEU"):
            src_input_ids = torch.stack([item['src_input_ids'] for item in batch]).to(device)
            labels = torch.stack([item['labels'] for item in batch]).to(device)
            src_mask, _ = create_masks(src_input_ids, labels, tokenizer.pad_token_id)
            src_mask = src_mask.to(device)

            # 使用贪心解码（逐步预测）
            memory = model.encode(src_input_ids, src_mask)
            ys = torch.ones(src_input_ids.size(0), 1).fill_(tokenizer.cls_token_id).type_as(src_input_ids)

            for i in range(max_len - 1):
                tgt_mask = torch.tril(torch.ones((1, ys.size(1), ys.size(1)), device=device)).bool()
                out = model.decode(memory, src_mask, ys, tgt_mask)
                prob = model.generator(out[:, -1])
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if (next_word == tokenizer.sep_token_id).all():
                    break

            # 计算 BLEU
            for pred, ref in zip(ys, labels):
                pred_tokens = tokenizer.convert_ids_to_tokens(pred.tolist(), skip_special_tokens=True)
                ref_tokens = tokenizer.convert_ids_to_tokens(ref.tolist(), skip_special_tokens=True)
                if len(ref_tokens) > 0 and len(pred_tokens) > 0:
                    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
                    total_bleu += bleu
                    count += 1

    return total_bleu / count if count > 0 else 0



# ===============================
# 训练阶段
# ===============================
def train(model, train_dataloader, val_dataloader, optimizer, criterion, pad_token_id, device, epochs=10):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            src_input_ids = torch.stack([item['src_input_ids'] for item in batch]).to(device)
            tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch]).to(device)
            labels = torch.stack([item['labels'] for item in batch]).to(device)

            src_mask, tgt_mask = create_masks(src_input_ids, tgt_input_ids, pad_token_id)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            # 首批打印设备一致性信息
            if epoch == 0 and batch_idx == 0:
                check_device_consistency(model, src_input_ids, tgt_input_ids, src_mask, tgt_mask, labels)

            optimizer.zero_grad()
            decoder_out = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)
            logits = model.generator(decoder_out)

            if labels.max() >= logits.size(-1):
                print(f"⚠️ 标签越界！labels.max={labels.max().item()} >= vocab_size={logits.size(-1)}，跳过该批次。")
                continue
            if torch.isnan(logits).any():
                print("⚠️ logits 出现 NaN，跳过该批次。")
                continue

            loss = criterion(logits.contiguous().view(-1, logits.size(-1)),
                             labels.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            non_pad_tokens = (labels != pad_token_id).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

        train_loss = total_loss / total_tokens if total_tokens > 0 else 0
        val_loss = evaluate(model, val_dataloader, criterion, pad_token_id, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../results/best_transformer_model.pth')

        print(f"Epoch {epoch+1} finished in {time.time()-start_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


# ===============================
# 绘制损失曲线
# ===============================
def draw_loss(data, title):
    plt.figure(figsize=(10, 8))
    x = torch.arange(1, len(data) + 1)
    plt.plot(x, data, '-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} Loss')
    path=f"../results/{title}.png"
    os.makedirs(os.path.dirname(path),exist_ok=True)
    plt.savefig(path)

def set_seed(seed=42):
    """固定随机种子以保证实验可复现"""
    import random, os, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 随机种子已设置为: {seed}")



# ===============================
# 主程序入口
# ===============================
def main():
    set_seed(42)
    # ✅ 指定 GPU
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#    print(f"tokenizer.vocab_size = {tokenizer.vocab_size}")
#    print(f"pad_token_id = {tokenizer.pad_token_id}")

    CONFIG['src_vocab_size'] = tokenizer.vocab_size
    CONFIG['tgt_vocab_size'] = tokenizer.vocab_size

    # 数据集
    train_dataset, val_dataset, test_dataset = create_translation_datasets(
        '../train.de', '../train.en', tokenizer,
        val_ratio=0.1, test_ratio=0.1, max_length=64
    )

    train_dataloader = create_translation_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = create_translation_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = create_translation_dataloader(test_dataset, batch_size=64, shuffle=False)

    # 模型构建
    true_vocab_size = tokenizer.vocab_size
#    print(f"✅ 正确的词汇表大小: {true_vocab_size}")

    model = make_model(
        source_vocab=true_vocab_size,
        target_vocab=true_vocab_size,
        d_model=CONFIG['d_model'],
        head=CONFIG['nhead'],
        d_ff=CONFIG['dim_feedforward'],
        N=CONFIG['num_encoder_layers'],
        dropout=CONFIG['dropout']
    )

    # ✅ 替换嵌入层并迁移到 device
    model.src_embed = nn.Sequential(
        nn.Embedding(true_vocab_size, CONFIG['d_model']),
        model.src_embed[1]
    ).to(device)

    model.tgt_embed = nn.Sequential(
        nn.Embedding(true_vocab_size, CONFIG['d_model']),
        model.tgt_embed[1]
    ).to(device)

    # ✅ generator 也放到 device
    model.generator = nn.Linear(CONFIG['d_model'], true_vocab_size).to(device)

    model = model.to(device)
#    print(f"✅ 模型输出层维度已修正为: {model.generator.out_features}")

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ===============================
    # 开始训练
    # ===============================
    print("Starting training...")
    train_losses, val_losses = train(
        model, train_dataloader, val_dataloader,
        optimizer, criterion, tokenizer.pad_token_id, device, epochs=25
    )

    # ===============================
    # 测试阶段
    # ===============================
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('../results/best_transformer_model.pth'))
    test_loss = evaluate(model, test_dataloader, criterion, tokenizer.pad_token_id, device)
    print(f"Test Loss: {test_loss:.4f}, Perplexity: {math.exp(test_loss):.4f}")
    bleu_score = calculate_bleu(model, test_dataloader, tokenizer, device)
    print(f"🟢 BLEU Score: {bleu_score:.4f}")

    draw_loss(train_losses, 'Train')
    draw_loss(val_losses, 'Val')


if __name__ == "__main__":
    main()
