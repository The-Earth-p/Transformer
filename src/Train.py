from TransformerModel import make_model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

CONFIG={
    "source_vocab_size":10000,
    "target_vocab_size":10000,
    "layer_nums":6,
    "head_nums":8,
    "embedding_dim":512,
    "ffn_dim":1024,
    "dropout":0.1
}


def evaluate_perEpoch(model,val_dataloader,loss_function):
    model.eval()
    val_loss=0
    batch_num=0
    with torch.no_grad():
        for input,target in val_dataloader:
            output=model(input)
            loss=loss_function(output,target)
            val_loss+=loss.item()
            batch_num+=1
    return val_loss/batch_num

def train(model,train_dataloader,val_dataloader,optimizer,loss_function,num_epoches):
    train_losses,val_losses=[],[]
    model.train()
    for epoch in range(num_epoches):
        train_loss=0
        batch_num=0
        for input,target in train_dataloader:
            optimizer.zero_grad()
            output=model(input)
            loss=loss_function(output,target)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            batch_num+=1
        train_losses.append(train_loss/batch_num)
        val_loss=evaluate_perEpoch(model,val_dataloader,loss_function)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epoches}, Train Loss: {train_loss/batch_num:.4f}, Val Loss: {val_loss:.4f}")
    return train_losses,val_losses

if __name__=="__main__":
    model=make_model(source_vocab=CONFIG["source_vocab_size"],target_vocab=CONFIG["target_vocab_size"],N=CONFIG["layer_nums"],
                     d_model=CONFIG["embedding_dim"],d_ff=CONFIG["ffn_dim"],head=CONFIG["head_nums"],dropout=CONFIG["dropout"])
