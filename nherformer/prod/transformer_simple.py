import torch
import torch.nn as nn
import torch.nn.functional as F
from nherformer.prod import config


# --- hyperparameters ---
block_size = 16
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_iters = 5000
eval_iters = 300
n_embd = 64 * 2
n_head = 8
n_layer = 8
dropout = 0.1
vocab_size = 120301230123

if device.type == "cuda":
    print("GPU available, using CUDA")
else:
    print("GPU not available, using CPU")
# --- setup ---

with open("../../data/input.txt", "r") as f:
    text = f.read()


chars = sorted(list(set(text)))

# Overwriting the vocab size to be the number of unique characters
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(set(chars))}
itos = {i: s for i, s in enumerate(set(chars))}
encode = lambda s: [stoi[c] for c in s]  # encoder: string -> list of indices (tokens)
decode = lambda s: "".join(
    [itos[i] for i in s]
)  # decoder: list of indices (tokens) -> string


data = torch.tensor(encode(text), dtype=torch.long)

# Let's now split up the data into train and validation sets
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# --- model ---


class Head(nn.Module):
    """A single head in a multi-headed attention layer"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention
        weights = (
            q @ k.transpose(-2, -1) * C ** (-0.5)
        )  # This is the scaled dot product, we scale because we don't want the dot product to be too large
        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # Mask out the upper triangular part of the matrix so we don't attend to future tokens
        weights = F.softmax(
            weights, dim=-1
        )  # Turn into probabilities of attension to each token
        weights = self.dropout(
            weights
        )  # Apply dropout because we don't want to overfit to the attention weights

        v = self.value(x)
        out = (
            weights @ v
        )  # Apply attention weights to the values (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multi-headed attention layer with multiple heads in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # Concatenate the outputs of each head
        out = self.dropout(
            self.proj(out)
        )  # Apply a linear projection to the concatenated outputs
        return out


class FeedForward(nn.Module):
    """Feed-forward layer (two linear layers with a ReLU in between)"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A transformer block: Communication followed by computation"""

    def __init__(self, n_emd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NherfLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Final linear layer

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.embedding(x)  # Embed the tokens
        pos_emb = self.position_embedding(
            torch.arange(T, device=device)
        )  # Embed the positions
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # B = batch, T = time, C = channels
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, max_new_tokens=100):
        for _ in range(max_new_tokens):
            # crop x to the last block_size tokens
            x_cond = x if x.shape[-1] <= block_size else x[:, -block_size:]
            # Get predictions for the next token
            logits, _ = self(x_cond)
            # Focus on the last token
            logits = logits[:, -1, :]  # (B, C)
            # Turn into probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution and get the most probable sample
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append to the sequence and continue
            x = torch.cat([x, next_token], dim=-1)

        return x


model = NherfLanguageModel()
model = model.to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for iter in range(max_iters):
    if iter % 500 == 0:
        print(f"Started training iteration {iter}")
        losses = estimate_loss()
        print(f"Step {iter}: Train loss {losses['train']}, Val loss {losses['val']}")

    x_batch, y_batch = get_batch("train")

    logits, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros(1, 1, dtype=torch.long, device=device)
print("\n=====\n\n")
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
