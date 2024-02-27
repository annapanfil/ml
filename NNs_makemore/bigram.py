import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64 # how many independent sequences will we process in paralell
block_size = 256 # max context_length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4 # for bigger models, lower learning rate is better
eval_iters = 200
n_embed = 384 
n_head = 4
# 384 // 6 = 64
n_layer = 6
dropout = 0.2

################################################
print("Running on", device)

# READ DATA
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# CREATE MAPPING AND ENCODE DATA
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda word: [stoi[c] for c in word] 
decode = lambda word: "".join([itos[c] for c in word])

data = torch.tensor(encode(text), dtype=torch.long)

# SPLITS
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# DATA LOADING
def get_batch(split):
    # generate a random chunk of inputs x and targets y
    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # beginning of the context
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class FeedForward(nn.Module):
    """simple linear layer with nonlinearity"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # 4 comes from the paper "Attention is all you need"
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))) # not a parameter, but buffer

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, H)
        q = self.query(x) # (B, T, H)
        # compute attention scores (affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei) # prevent some nodes from communicating
        # weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        # embedding size and number of heads
        super().__init__()
        head_size = n_embed // n_head
        
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed) # normalization before attention (not like in paper, but more common now)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection to help learn big network
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size) # language model

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B, T, C=n_emb)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) #(B, T, vocab_size) – (batch, time, channel)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # pytorch expects C in 2nd dimention
            targets = targets.view(B*T) # or -1
            loss = F.cross_entropy(logits, targets) 
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to fit the block_size
            logits, _ = self(idx_cond) # feed all history – forward
            # focus only on the last time step (last elem in time dimention)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append 
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out

# TRAINING BIGRAM MODEL
m = BigramLanguageModel()
m = m.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for i in range(max_iters):
    # average loss from multiple barches
    if i % eval_interval == 0:
        losses = estimate_loss(m)
        print(f"step {i}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch("test")
    # forward
    logits, loss = m(xb, yb)
    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # update
    optimizer.step()

# GENERATE
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
