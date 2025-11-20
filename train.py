import torch
import torch.nn as nn
from torch.nn import functional as F
from model import PSM
from tqdm import tqdm
import time
import math
import os
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED CONFIGURATION ---
class Config:
    # Data & Model
    FILE_PATH = 'my_writings.txt'
    VOCAB_SIZE = None  # Will be set automatically
    BLOCK_SIZE = 256  # Increased context window
    N_EMBD = 384  # Embedding dimension
    N_LAYER = 6   # Number of transformer layers
    N_HEAD = 6    # Number of attention heads
    
    # Training
    BATCH_SIZE = 64  # Increased batch size
    MAX_STEPS = 5000  # More training steps
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 500  # Learning rate warmup
    MIN_LR = 1e-5      # Minimum learning rate
    
    # Regularization
    DROPOUT = 0.1
    WEIGHT_DECAY = 0.1
    
    # Evaluation & Saving
    EVAL_INTERVAL = 250
    SAVE_INTERVAL = 1000
    GENERATION_LENGTH = 200
    
    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    COMPILE_MODEL = True  # Use torch.compile for speed (PyTorch 2.0+)
    
    def __init__(self):
        pass

config = Config()

# --- ENHANCED DATA LOADING ---
print("Reading training data...")
try:
    with open(config.FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Loaded {len(text):,} characters")
except FileNotFoundError:
    print(f"Error: File {config.FILE_PATH} not found!")
    exit(1)

# --- ADVANCED TOKENIZER ---
chars = sorted(list(set(text)))
config.VOCAB_SIZE = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print(f"Vocabulary size: {config.VOCAB_SIZE}")

# --- OPTIMIZED DATA LOADING ---
print("Preparing training data...")
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

# --- LEARNING RATE SCHEDULER ---
def get_lr(step):
    if step < config.WARMUP_STEPS:
        return config.LEARNING_RATE * (step + 1) / config.WARMUP_STEPS
    progress = (step - config.WARMUP_STEPS) / (config.MAX_STEPS - config.WARMUP_STEPS)
    return config.MIN_LR + 0.5 * (config.LEARNING_RATE - config.MIN_LR) * (1 + math.cos(math.pi * progress))

# --- ENHANCED MODEL INITIALIZATION ---
print(f"Initializing PSM model on {config.DEVICE.upper()}...")
model = PSM(
    vocab_size=config.VOCAB_SIZE,
    n_embd=config.N_EMBD,
    n_layer=config.N_LAYER,
    n_head=config.N_HEAD,
    block_size=config.BLOCK_SIZE,
    dropout=config.DROPOUT
)
model = model.to(config.DEVICE)

# Model compilation for faster training (PyTorch 2.0+)
if config.COMPILE_MODEL and hasattr(torch, 'compile'):
    try:
        model = torch.compile(model)
        print("Model compiled for optimized performance")
    except Exception as e:
        print(f"Model compilation failed: {e}")

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

# --- ADVANCED OPTIMIZER ---
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    betas=(0.9, 0.95)
)

# --- TENSORBOARD LOGGING ---
writer = SummaryWriter(log_dir='runs/psm_training')

# --- ENHANCED GENERATION FUNCTION ---
def generate_thought(model, prompt=" ", temperature=0.8, top_k=40):
    model.eval()
    with torch.no_grad():
        context = torch.tensor(encode(prompt), dtype=torch.long, device=config.DEVICE).unsqueeze(0)
        
        for _ in range(config.GENERATION_LENGTH):
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Stop if we generate a newline (end of thought)
            if idx_next.item() == encode('\n')[0]:
                break
                
            context = torch.cat((context, idx_next), dim=1)
            
    model.train()
    return decode(context[0].tolist())

# --- TRAINING LOOP WITH VALIDATION ---
print("\nStarting PSM training...")
start_time = time.time()
best_val_loss = float('inf')
global_step = 0

# Create checkpoint directory
os.makedirs('checkpoints', exist_ok=True)

progress_bar = tqdm(range(config.MAX_STEPS), desc="Training", unit="step")

for step in progress_bar:
    # Learning rate scheduling
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Get batch
    x, y = get_batch('train')
    
    # Forward pass
    logits, loss = model(x, y)
    
    # Backward pass with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Log training loss
    writer.add_scalar('Loss/train', loss.item(), step)
    writer.add_scalar('LearningRate', lr, step)
    
    # Update progress bar
    progress_bar.set_postfix({
        "Loss": f"{loss.item():.4f}",
        "LR": f"{lr:.2e}"
    })
    
    # Validation and generation
    if step % config.EVAL_INTERVAL == 0 or step == config.MAX_STEPS - 1:
        model.eval()
        with torch.no_grad():
            # Calculate validation loss
            val_losses = []
            for _ in range(100):  # Use multiple batches for stable validation
                x_val, y_val = get_batch('val')
                _, val_loss = model(x_val, y_val)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)
            writer.add_scalar('Loss/val', avg_val_loss, step)
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'checkpoints/psm_best.pth')
            
            # Generate sample text
            thought = generate_thought(model, temperature=0.8)
            writer.add_text('Generation', thought, step)
            
            print(f"\nStep {step}:")
            print(f"Train Loss: {loss.item():.4f}, Val Loss: {avg_val_loss:.4f}")
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
torch.save(model.state_dict(), "psm_final.pth")
print("Final model saved as psm_final.pth")

# Save tokenizer metadata
tokenizer_data = {
    'stoi': stoi,
    'itos': itos,
    'vocab_size': config.VOCAB_SIZE,
    'config': config.__dict__
}
torch.save(tokenizer_data, 'tokenizer.pth')
print("Tokenizer metadata saved")

writer.close()
print("Training complete!")