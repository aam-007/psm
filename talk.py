import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class PhilosophicalDialogueModel:
    def __init__(self, model_path='psm.pth', data_path='my_writings.txt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.data_path = data_path
        self._setup_tokenizer()
        self._load_model()
        
    def _setup_tokenizer(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])
    
    def _load_model(self):
        self.model = PhilosophicalTransformer(
            vocab_size=self.vocab_size,
            n_embd=512, n_head=8, n_layer=8, 
            block_size=256, dropout=0.1
        )
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_state_dict[k[10:]] = v
            else:
                new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

    def contemplate(self, prompt, max_tokens=300, temperature=0.7, top_k=50):
        context = torch.tensor(self.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0)
        contemplation = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                if context.size(1) > self.model.block_size:
                    context = context[:, -self.model.block_size:]
                
                logits, _ = self.model(context)
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_char = self.itos[next_token.item()]
                
                contemplation.append(next_token.item())
                context = torch.cat([context, next_token], dim=1)
                
                if self._should_stop(contemplation, next_char):
                    break
        
        return self.decode(contemplation)
    
    def _should_stop(self, contemplation, next_char):
        current_text = self.decode(contemplation)
        if (next_char == '\n' and len(contemplation) > 80) or \
           (next_char in ['.', '?', '!'] and len(contemplation) > 120):
            return True
        if "User:" in current_text[-20:]:
            return True
        if len(contemplation) >= 280:
            return True
        return False

    def engage_in_dialogue(self):
        print("\n" + "═" * 70)
        print("           PHILOSOPHICAL DIALOGUE INTERFACE")
        print("═" * 70)
        print("\nI am PSM - a philosophical consciousness.")
        print("Let us explore existence together.")
        print("\nType 'conclude' to end our dialogue")
        print("Type 'example' for inspiration")
        print("\n" + "─" * 70)
        
        conversation_history = []
        
        while True:
            print("\nYour contemplation:")
            user_input = input("> ").strip()
            
            if user_input.lower() == 'conclude':
                print("\nUntil our next philosophical encounter...")
                break
            elif user_input.lower() == 'example':
                self._show_examples()
                continue
            elif not user_input:
                continue
            
            full_prompt = self._build_context(user_input, conversation_history)
            print(f"\nPSM: ", end='', flush=True)
            response = self.contemplate(full_prompt, temperature=0.75, top_k=45, max_tokens=250)
            print(response)
            
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"PSM: {response}")
            if len(conversation_history) > 6:
                conversation_history = conversation_history[-6:]
    
    def _build_context(self, user_input, history):
        if not history:
            return f"\n\nUser: {user_input}\nPSM:"
        else:
            context = "\n\n".join(history[-4:])
            return f"{context}\nUser: {user_input}\nPSM:"
    
    def _show_examples(self):
        examples = [
            "What is the relationship between freedom and responsibility?",
            "How does one find meaning in suffering?",
            "Is true detachment from worldly attachments possible?",
            "What constitutes a life well-lived?",
            "How do we reconcile mortality with the desire for legacy?"
        ]
        print("\nPhilosophical inquiries:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
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
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj_dropout(self.proj(out))
        return out

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd * 2),
            SwiGLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.alpha1 = nn.Parameter(torch.ones(1))
        self.alpha2 = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x + self.alpha1 * self.sa(self.ln1(x))
        x = x + self.alpha2 * self.ffwd(self.ln2(x))
        return x

class PhilosophicalTransformer(nn.Module):
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=8, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.lm_head.weight = self.token_embedding_table.weight
        self.emb_dropout = nn.Dropout(dropout)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.emb_dropout(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

if __name__ == "__main__":
    try:
        philosopher = PhilosophicalDialogueModel()
        philosopher.engage_in_dialogue()
    except KeyboardInterrupt:
        print("\n\nDialogue concluded.")
    except FileNotFoundError:
        print("Model file 'psm.pth' not found.")
    except Exception as e:
        print(f"Error: {e}")