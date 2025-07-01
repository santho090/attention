import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import time
import numpy as np
from typing import List, Tuple, Optional
from transformer import Transformer, create_transformer_model


class TranslationDataset(Dataset):
    """
    Dataset for machine translation tasks
    """
    
    def __init__(self, src_sentences: List[List[int]], tgt_sentences: List[List[int]], 
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0, max_len: int = 100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src = self.src_sentences[idx]
        tgt = self.tgt_sentences[idx]
        
        # Truncate if too long
        src = src[:self.max_len]
        tgt = tgt[:self.max_len]
        
        return torch.tensor(src), torch.tensor(tgt)


def collate_fn(batch, src_pad_idx: int = 0, tgt_pad_idx: int = 0):
    """
    Collate function for DataLoader to handle variable length sequences
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    
    return src_batch, tgt_batch


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss as mentioned in the paper
    """
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = pred.shape
        
        # Reshape for easier computation
        pred = pred.reshape(-1, vocab_size)
        target = target.reshape(-1)
        
        # Create one-hot encoding
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 2))  # -2 for true class and pad
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        
        # Mask out padding tokens
        mask = (target != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)
        
        # Compute KL divergence
        kl_div = F.kl_div(F.log_softmax(pred, dim=1), true_dist, reduction='none')
        loss = kl_div.sum(dim=1)
        
        return loss[mask].mean()


class NoamOptimizer:
    """
    Noam learning rate scheduler as described in the paper
    """
    
    def __init__(self, model_size: int, warmup_steps: int, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.model_size ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.optimizer.step()
        
    def zero_grad(self):
        self.optimizer.zero_grad()


def generate_dummy_data(src_vocab_size: int, tgt_vocab_size: int, 
                       num_samples: int = 1000, max_len: int = 50) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Generate dummy translation data for demonstration
    """
    src_sentences = []
    tgt_sentences = []
    
    for _ in range(num_samples):
        # Random source sentence length
        src_len = np.random.randint(5, max_len)
        tgt_len = np.random.randint(5, max_len)
        
        # Generate random sentences (avoiding pad token 0)
        src_sentence = np.random.randint(1, src_vocab_size, src_len).tolist()
        tgt_sentence = np.random.randint(1, tgt_vocab_size, tgt_len).tolist()
        
        src_sentences.append(src_sentence)
        tgt_sentences.append(tgt_sentence)
    
    return src_sentences, tgt_sentences


def train_epoch(model: Transformer, dataloader: DataLoader, optimizer: NoamOptimizer, 
                criterion: nn.Module, device: torch.device) -> float:
    """
    Train the model for one epoch
    """
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        
        # Prepare target input and output
        tgt_input = tgt[:, :-1]  # All tokens except last
        tgt_output = tgt[:, 1:]  # All tokens except first
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
    
    return total_loss / num_batches


def evaluate(model: Transformer, dataloader: DataLoader, criterion: nn.Module, 
             device: torch.device) -> float:
    """
    Evaluate the model
    """
    model.eval()
    total_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Prepare target input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Compute loss
            loss = criterion(output, tgt_output)
            total_loss += loss.item()
    
    return total_loss / num_batches


def greedy_decode(model: Transformer, src: torch.Tensor, src_mask: torch.Tensor,
                  max_len: int, start_symbol: int, device: torch.device) -> torch.Tensor:
    """
    Greedy decoding for inference
    """
    model.eval()
    
    # Encode source
    encoder_output = model.encode(src, src_mask)
    
    # Initialize target with start symbol
    tgt = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    
    for i in range(max_len - 1):
        # Create target mask
        tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
        
        # Decode
        output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Get next token
        next_token = output[:, -1, :].argmax(dim=-1)
        
        # Append to target
        tgt = torch.cat([tgt, next_token.unsqueeze(1)], dim=1)
        
        # Stop if end token is generated (assuming end token is 2)
        if next_token.item() == 2:
            break
    
    return tgt


def main():
    """
    Main training function
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_len = 100
    batch_size = 32
    num_epochs = 10
    warmup_steps = 4000
    
    # Generate dummy data
    print("Generating dummy data...")
    src_sentences, tgt_sentences = generate_dummy_data(src_vocab_size, tgt_vocab_size, 
                                                      num_samples=5000, max_len=max_len)
    
    # Split data
    train_size = int(0.8 * len(src_sentences))
    train_src, val_src = src_sentences[:train_size], src_sentences[train_size:]
    train_tgt, val_tgt = tgt_sentences[:train_size], tgt_sentences[train_size:]
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, max_len=max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, max_len=max_len)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=lambda x: collate_fn(x, 0, 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda x: collate_fn(x, 0, 0))
    
    # Create model
    print("Creating model...")
    model = create_transformer_model(src_vocab_size, tgt_vocab_size, d_model, 
                                   n_heads, n_layers, d_ff).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamOptimizer(d_model, warmup_steps, optimizer)
    
    # Create loss function
    criterion = LabelSmoothingLoss(tgt_vocab_size, smoothing=0.1, pad_idx=0)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, scheduler, criterion, device)
        train_time = time.time() - start_time
        
        # Evaluate
        start_time = time.time()
        val_loss = evaluate(model, val_loader, criterion, device)
        val_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')
            print("Saved best model!")
    
    print("Training completed!")
    
    # Example inference
    print("\nExample inference:")
    model.eval()
    
    # Create a dummy source sentence
    src_example = torch.randint(1, src_vocab_size, (1, 10)).to(device)
    src_mask = model.create_padding_mask(src_example, 0)
    
    # Generate translation
    with torch.no_grad():
        translation = greedy_decode(model, src_example, src_mask, max_len=20, 
                                  start_symbol=1, device=device)
    
    print(f"Source: {src_example.squeeze().tolist()}")
    print(f"Translation: {translation.squeeze().tolist()}")


if __name__ == "__main__":
    main()