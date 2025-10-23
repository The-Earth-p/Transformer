import torch
import torch.nn as nn
import torch.optim as optim
import math
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
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
    print("=== 设备一致性检查 ===")
    print(f"模型参数设备: {next(model.parameters()).device}")
    print(f"源输入设备: {src_input_ids.device}")
    print(f"目标输入设备: {tgt_input_ids.device}")
    print(f"源掩码设备: {src_mask.device}")
    print(f"目标掩码设备: {tgt_mask.device}")
    print(f"标签设备: {labels.device}")
    devices = [src_input_ids.device, tgt_input_ids.device, src_mask.device, tgt_mask.device, labels.device]
    if all(d == devices[0] for d in devices):
        print("✅ 所有张量在相同设备上")
    else:
        print("❌ 张量设备不一致！")


# ===============================
# 验证与测试阶段 （注意：apply generator to get logits）
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

            src_mask, tgt_mask = create_masks(src_input_ids, tgt_input_ids, pad_token_id)
            decoder_out = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)  # (B, T, d_model)
            logits = model.generator(decoder_out)  # (B, T, vocab_size)

            # compute loss: CrossEntropyLoss expects (N, C) logits and (N,) targets
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            non_pad_tokens = (labels != pad_token_id).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0


# ===============================
# 训练阶段 （注意：apply generator to get logits）
# ===============================
def train(model, train_dataloader, val_dataloader, optimizer, criterion, pad_token_id, device, epochs=10):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            src_input_ids = torch.stack([item['src_input_ids'] for item in batch]).to(device)
            tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch]).to(device)
            labels = torch.stack([item['labels'] for item in batch]).to(device)

            src_mask, tgt_mask = create_masks(src_input_ids, tgt_input_ids, pad_token_id)

            optimizer.zero_grad()

            decoder_out = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)  # (B, T, d_model)
            logits = model.generator(decoder_out)  # (B, T, vocab_size)

            # 🔍 调试打印：使用 logits 的最后维度（vocab size）
            print(f"labels.max={labels.max().item()} | logits vocab_size={logits.size(-1)}")

            # 防止越界或 NaN
            if labels.max() >= logits.size(-1):
                print(f"⚠️ 标签越界！labels.max={labels.max().item()} >= vocab_size={logits.size(-1)}，跳过该批次。")
                continue
            if torch.isnan(logits).any():
                print("⚠️ logits 出现 NaN，跳过该批次。")
                continue

            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
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
            torch.save(model.state_dict(), 'best_transformer_model.pth')

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
    plt.savefig(f"../result/{title}.png")


# ===============================
# 主程序入口
# ===============================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()

    # ✅ 保证 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print(f"tokenizer.vocab_size = {tokenizer.vocab_size}")
    print(f"pad_token_id = {tokenizer.pad_token_id}")

    CONFIG['src_vocab_size'] = tokenizer.vocab_size
    CONFIG['tgt_vocab_size'] = tokenizer.vocab_size

    # 创建数据集
    train_dataset, val_dataset, test_dataset = create_translation_datasets(
        '../train.de', '../train.en', tokenizer,
        val_ratio=0.1, test_ratio=0.1, max_length=128
    )

    train_dataloader = create_translation_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = create_translation_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = create_translation_dataloader(test_dataset, batch_size=32, shuffle=False)

    # ====================================
    # ✅ 创建模型并强制匹配 tokenizer
    # ====================================
    true_vocab_size = tokenizer.vocab_size
    print(f"✅ 正确的词汇表大小: {true_vocab_size}")

    model = make_model(
        source_vocab=true_vocab_size,
        target_vocab=true_vocab_size,
        d_model=CONFIG['d_model'],
        head=CONFIG['nhead'],
        d_ff=CONFIG['dim_feedforward'],
        N=CONFIG['num_encoder_layers'],
        dropout=CONFIG['dropout']
    )

    # ✅ 强制替换嵌入层（保持位置编码）并移动到 device
    model.src_embed = nn.Sequential(
        nn.Embedding(true_vocab_size, CONFIG['d_model']),
        model.src_embed[1]
    ).to(device)

    model.tgt_embed = nn.Sequential(
        nn.Embedding(true_vocab_size, CONFIG['d_model']),
        model.tgt_embed[1]
    ).to(device)

    # ✅ generator: 只用线性层产生 logits（不要 log_softmax），交给 CrossEntropyLoss 处理
    model.generator = nn.Linear(CONFIG['d_model'], true_vocab_size).to(device)

    # 再确保整体都在 device 上
    model = model.to(device)
    print(f"✅ 模型输出层维度已修正为: {model.generator.out_features}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # ===============================
    # 开始训练
    # ===============================
    print("Starting training...")
    train_losses, val_losses = train(
        model, train_dataloader, val_dataloader,
        optimizer, criterion, tokenizer.pad_token_id, device, epochs=10
    )

    # 测试
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    test_loss = evaluate(model, test_dataloader, criterion, tokenizer.pad_token_id, device)
    print(f"Test Loss: {test_loss:.4f}, Perplexity: {math.exp(test_loss):.4f}")

    draw_loss(train_losses, 'Train')
    draw_loss(val_losses, 'Val')


if __name__ == "__main__":
    main()
