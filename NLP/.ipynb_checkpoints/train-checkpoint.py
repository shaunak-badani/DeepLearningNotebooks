import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

## Hyperparameters
block_size = 256
batch_size = 64
max_iters = 30000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device :", device)
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------

# All characters in our text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Mapping from characters to integers and back
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    encoded_string = []
    for c in s:
        encoded_string.append(stoi[c])
    return encoded_string

def decode(s):
    decoded_string = []
    for c in s:
        decoded_string.append(chars[c])
    return ''.join(decoded_string)

data = torch.tensor(encode(text), dtype = torch.long)

# Splitting data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    # ix are indices
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y

channel_size = vocab_size
torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, Hs)
        q = self.query(x) # (B, T, Hs)
        v = self.value(x) # (B, T, Hs)

        wei = q @ k.transpose(-2, -1) * self.head_size ** (-0.5) # (B, T, Hs) @ (B, Hs, T) -> (B, T, T)
        wei = F.softmax(wei, dim = -1)
        out = wei @ v
        return out

class MaskedHead(nn.Module):
    """One head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, Hs)
        q = self.query(x) # (B, T, Hs)
        v = self.value(x) # (B, T, Hs)

        wei = q @ k.transpose(-2, -1) * self.head_size ** (-0.5) # (B, T, Hs) @ (B, Hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril == 0, float('-inf'))
        # wei = self.dropout(wei)
        wei = F.softmax(wei, dim = -1)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([MaskedHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """ A simple linear layer followed by non linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, num_embd, n_head):
        super().__init__()
        head_size = num_embd // n_head
        self.mh = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(num_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mh(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # dimensions (Batch_size, time, Channels = channel_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        # For each token, we get a (channel_size) vector which is the probability of the next token being i. i belongs to [0, vocab_size)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for iter_n in range(max_iters):
    if iter_n % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter_n}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch("train")
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
starting_token = torch.zeros((1,block_size), dtype = torch.long)
generated_gibberish = m.generate(starting_token, max_new_tokens = 1000)
print(decode(generated_gibberish[0].tolist()))
