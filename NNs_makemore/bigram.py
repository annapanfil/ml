import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1337)
batch_size = 32 # how many independent sequences will we process in paralell
block_size = 8 # max context_length
max_iters = 1000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32

################################################

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

# BASELINE MODEL
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx) # (B, T, C=n_emb)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C)
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
        for _ in range(max_new_tokens):
            logits, _ = self(idx) # feed all history to the – not really needed here
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
