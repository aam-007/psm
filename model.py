import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import time
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED CONFIGURATION ---
class Config:
    # Model Architecture
    VOCAB_SIZE = None  # Will be set automatically
    BLOCK_SIZE = 256   # Increased context window
    N_EMBD = 512       # Increased embedding dimension
    N_HEAD = 8         # More attention heads
    N_LAYER = 8        # More layers
    DROPOUT = 0.1      # Optimized dropout
    
    # Training
    BATCH_SIZE = 32
    GRAD_ACCUM_STEPS = 4  # Gradient accumulation
    MAX_STEPS = 5000
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 1000
    MIN_LR = 1e-5
    WEIGHT_DECAY = 0.1
    
    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    COMPILE_MODEL = True
    
    # Evaluation
    EVAL_INTERVAL = 250
    SAVE_INTERVAL = 1000
    GENERATION_LENGTH = 200

config = Config()

# --- ENHANCED MODEL ARCHITECTURE ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.query = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.value = nn.Linear(config.N_EMBD, head_size, bias=False)
        self.attn_dropout = nn.Dropout(config.DROPOUT)
        self.resid_dropout = nn.Dropout(config.DROPOUT)
        
        self.register_buffer('tril', torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, T, C = x.shape
        
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2, -1) * (self.scale / math.sqrt(self.head_size))
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        out = self.resid_dropout(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.proj_dropout = nn.Dropout(config.DROPOUT)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj_dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd * 2),
            SwiGLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(config.DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x + self.alpha1 * self.sa(self.ln1(x))
        x = x + self.alpha2 * self.ffwd(self.ln2(x))
        return x

class PSM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.block_size = config.BLOCK_SIZE
        self.vocab_size = vocab_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBD)
        self.position_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBD)
        
        self.blocks = nn.Sequential(*[
            Block(config.N_EMBD, n_head=config.N_HEAD) for _ in range(config.N_LAYER)
        ])
        
        self.ln_f = nn.LayerNorm(config.N_EMBD)
        self.lm_head = nn.Linear(config.N_EMBD, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight  # Weight tying
        
        self.emb_dropout = nn.Dropout(config.DROPOUT)
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block size {self.block_size}"
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.emb_dropout(tok_emb + pos_emb)
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# --- DATA LOADING AND TRAINING SETUP ---
print("Loading training data...")
try:
    with open('my_writings.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")
except FileNotFoundError:
    print("Error: my_writings.txt not found!")
    exit(1)

# Build tokenizer
chars = sorted(list(set(text)))
config.VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print(f"Vocabulary size: {config.VOCAB_SIZE}")

# Prepare data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i:i+config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+config.BLOCK_SIZE+1] for i in ix])
    return x.to(config.DEVICE), y.to(config.DEVICE)

# Learning rate scheduler
def get_lr(step):
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / config.WARMUP_STEPS
    progress = (step - config.WARMUP_STEPS) / (config.MAX_STEPS - config.WARMUP_STEPS)
    return config.MIN_LR + 0.5 * (config.LEARNING_RATE - config.MIN_LR) * (1 + math.cos(math.pi * progress))

# Initialize model
print(f"Initializing enhanced PSM model on {config.DEVICE.upper()}...")
model = PSM(config.VOCAB_SIZE)
model = model.to(config.DEVICE)

# Model compilation
if config.COMPILE_MODEL and hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
        print("Model compiled for optimized performance")
    except Exception as e:
        print(f"Model compilation failed: {e}")

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    betas=(0.9, 0.95)
)

# TensorBoard
writer = SummaryWriter(log_dir='runs/psm_enhanced')

# Generation function
def generate_thought(model, prompt=" ", temperature=0.8, top_k=40):
    model.eval()
    with torch.no_grad():
        context = torch.tensor(encode(prompt), dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        
        for _ in range(config.GENERATION_LENGTH):
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next.item() == encode('\n')[0]:
                break
                
            context = torch.cat((context, idx_next), dim=1)
            
    model.train()
    return decode(context[0].tolist())

# --- TRAINING LOOP ---
print("\nStarting enhanced PSM training...")
start_time = time.time()
best_val_loss = float('inf')
global_step = 0

os.makedirs('checkpoints', exist_ok=True)

progress_bar = tqdm(range(config.MAX_STEPS), desc="Training", unit="step")

for step in progress_bar:
    # Learning rate scheduling
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Gradient accumulation
    for micro_step in range(config.GRAD_ACCUM_STEPS):
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss = loss / config.GRAD_ACCUM_STEPS
        loss.backward()
    
    # Gradient clipping and update
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    # Logging
    writer.add_scalar('Loss/train', loss.item() * config.GRAD_ACCUM_STEPS, step)
    writer.add_scalar('LearningRate', lr, step)
    
    progress_bar.set_postfix({
        "Loss": f"{loss.item() * config.GRAD_ACCUM_STEPS:.4f}",
        "LR": f"{lr:.2e}"
    })
    
    # Validation and generation
    if step % config.EVAL_INTERVAL == 0 or step == config.MAX_STEPS - 1:
        model.eval()
        with torch.no_grad():
            # Validation loss
            val_losses = []
            for _ in range(100):
                x_val, y_val = get_batch('val')
                _, val_loss = model(x_val, y_val)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            writer.add_scalar('Loss/val', avg_val_loss, step)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'checkpoints/psm_best.pth')
            
            # Generate sample
            thought = generate_thought(model, temperature=0.8)
            writer.add_text('Generation', thought, step)
            
            print(f"\nStep {step}:")
            print(f"Train Loss: {loss.item() * config.GRAD_ACCUM_STEPS:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"PSM: '{thought}'")
        
        model.train()
    
    # Save checkpoint
    if step % config.SAVE_INTERVAL == 0 and step > 0:
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'config': config.__dict__
        }
        torch.save(checkpoint, f'checkpoints/psm_step_{step}.pth')
        print(f"Checkpoint saved at step {step}")

# --- FINALIZATION ---
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Save final model
torch.save(model.state_dict(), "psm_enhanced.pth")
print("Final model saved as psm_enhanced.pth")

# Save tokenizer
tokenizer_data = {
    'stoi': stoi,
    'itos': itos,
    'vocab_size': config.VOCAB_SIZE,
    'config': config.__dict__
}
torch.save(tokenizer_data, 'tokenizer_enhanced.pth')
print("Tokenizer metadata saved")

writer.close()
print("Enhanced training complete!")