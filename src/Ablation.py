# AblationTrain.py  —— 已针对你项目的 DataProcess.py 做好适配
import os
import random
import math
import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from TransformerModel import make_model
from DataProcess import get_tokenizer, create_translation_datasets, create_translation_dataloader

# -------- reproducibility --------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 seed={seed}")

# -------- utilities that stack the batch (your DataLoader returns list of dicts) --------
def batch_to_tensors(batch, device):
    # batch is a list of dicts as in your DataProcess.__getitem__
    src_input_ids = torch.stack([item['src_input_ids'] for item in batch]).to(device)
    tgt_input_ids = torch.stack([item['tgt_input_ids'] for item in batch]).to(device)
    labels = torch.stack([item['labels'] for item in batch]).to(device)
    return src_input_ids, tgt_input_ids, labels

# -------- evaluate loss (uses same style as Train.py) --------
def evaluate(model, dataloader, criterion, pad_token_id, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src_input_ids, tgt_input_ids, labels = batch_to_tensors(batch, device)
            src_mask, tgt_mask = create_masks_for_eval(src_input_ids, tgt_input_ids, pad_token_id, device)
            out = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)
            logits = model.generator(out)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            non_pad = (labels != pad_token_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    model.train()
    return total_loss / total_tokens if total_tokens > 0 else 0.0

# helper to create masks (reuse your DataProcess.create_masks semantics)
def create_masks_for_eval(src, tgt, pad_token_id, device):
    # reuse same mask construction as DataProcess.create_masks
    src_mask = (src != pad_token_id).unsqueeze(-2).to(device)
    tgt_mask = (tgt != pad_token_id).unsqueeze(-2).to(device, dtype=torch.bool)
    seq_len = tgt.size(1)
    nopeak = torch.tril(torch.ones((1, seq_len, seq_len), device=device, dtype=torch.bool))
    tgt_mask = tgt_mask & nopeak
    return src_mask, tgt_mask

# -------- BLEU calculation (greedy decode) --------
def calculate_bleu(model, dataloader, tokenizer, device, max_len=64):
    model.eval()
    smooth = SmoothingFunction().method1
    total_bleu = 0.0
    count = 0
    pad = tokenizer.pad_token_id
    bos = tokenizer.bos_token_id or tokenizer.cls_token_id
    eos = tokenizer.eos_token_id or tokenizer.sep_token_id

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating BLEU"):
            src_input_ids, _, labels = batch_to_tensors(batch, device)
            src_mask = (src_input_ids != pad).unsqueeze(-2).to(device)

            memory = model.encode(src_input_ids, src_mask)
            ys = torch.ones(src_input_ids.size(0), 1).fill_(bos).type_as(src_input_ids).to(device)

            for _ in range(max_len - 1):
                tgt_mask = torch.tril(torch.ones((1, ys.size(1), ys.size(1)), device=device)).bool()
                out = model.decode(memory, src_mask, ys, tgt_mask)
                prob = model.generator(out[:, -1])
                next_word = torch.argmax(prob, dim=-1).unsqueeze(1)
                ys = torch.cat([ys, next_word], dim=1)
                if (next_word == eos).all():
                    break

            # convert to tokens and compute sentence BLEU
            for pred_ids, ref_ids in zip(ys, labels):
                pred_tokens = tokenizer.convert_ids_to_tokens(pred_ids.tolist(), skip_special_tokens=True)
                ref_tokens = tokenizer.convert_ids_to_tokens(ref_ids.tolist(), skip_special_tokens=True)
                # remove pad/bos/eos tokens if present in returned tokens list
                # NLTK expects lists of tokens (strings)
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
                    total_bleu += bleu
                    count += 1

    return total_bleu / count if count > 0 else 0.0

# -------- single-model trainer (matches Train.py stacking semantics) --------
def train_one_model(model, train_loader, val_loader, criterion, tokenizer, device, epochs=5, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    pad = tokenizer.pad_token_id
    best_val = float('inf')
    os.makedirs("./results", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for batch in tqdm(train_loader, desc=f"Training epoch {epoch}"):
            src_input_ids, tgt_input_ids, labels = batch_to_tensors(batch, device)
            src_mask, tgt_mask = create_masks_for_eval(src_input_ids, tgt_input_ids, pad, device)

            optimizer.zero_grad()
            out = model(src_input_ids, tgt_input_ids, src_mask, tgt_mask)
            logits = model.generator(out)
            # compute loss against labels (already shifted when creating dataset)
            loss = criterion(logits.contiguous().view(-1, logits.size(-1)), labels.contiguous().view(-1))
            loss.backward()
            optimizer.step()

            non_pad = (labels != pad).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

        train_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        val_loss = evaluate(model, val_loader, criterion, pad, device)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "./results/best_model_temp.pth")

    model.load_state_dict(torch.load("./results/best_model_temp.pth", map_location=device))
    model.to(device)
    return model, best_val

# -------- main ablation loop (uses your DataProcess functions) --------
def run_ablation():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = get_tokenizer()
    # create datasets and dataloaders using your DataProcess functions
    train_ds, val_ds, test_ds = create_translation_datasets('../train.de', '../train.en', tokenizer, val_ratio=0.1, test_ratio=0.1, max_length=64)
    train_loader = create_translation_dataloader(train_ds, batch_size=64, shuffle=True)
    val_loader = create_translation_dataloader(val_ds, batch_size=64, shuffle=False)
    test_loader = create_translation_dataloader(test_ds, batch_size=64, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    experiments = {
        "baseline": {"head": 8, "use_pos_encoding": True},
        "no_pos": {"head": 8, "use_pos_encoding": False},
        "single_head": {"head": 1, "use_pos_encoding": True},
    }

    results = {}
    for name, cfg in experiments.items():
        print("\n" + "="*30)
        print("Experiment:", name)
        print("="*30)
        model = make_model(
            source_vocab=tokenizer.vocab_size,
            target_vocab=tokenizer.vocab_size,
            d_model=512,
            d_ff=2048,
            head=cfg["head"],
            dropout=0.1,
            use_pos_encoding=cfg["use_pos_encoding"]  # <-- make_model must accept this arg
        ).to(device)

        model, val_loss = train_one_model(model, train_loader, val_loader, criterion, tokenizer, device, epochs=25, lr=1e-4)
        test_loss = evaluate(model, test_loader, criterion, tokenizer.pad_token_id, device)
        bleu = calculate_bleu(model, test_loader, tokenizer, device)
        results[name] = {"val_loss": val_loss, "test_loss": test_loss, "BLEU": bleu}
        print(f"{name} -> ValLoss: {val_loss:.4f}, TestLoss: {test_loss:.4f}, BLEU: {bleu:.4f}")

    print("\nFinal results:")
    for name, r in results.items():
        print(f"{name:12s} | Val {r['val_loss']:.4f} | Test {r['test_loss']:.4f} | BLEU {r['BLEU']:.4f}")

if __name__ == "__main__":
    run_ablation()
