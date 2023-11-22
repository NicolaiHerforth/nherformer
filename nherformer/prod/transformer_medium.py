import torch
import torch.nn as nn
import torch.nn.functional as F
from nherformer.prod.config import MediumNherformerConfig as ModelConfig
from nherformer.prod.utils.helper_funcs import estimate_loss, get_batch, get_tokenizer


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) module.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm module.

        Parameters
        ----------
        dim : int
            The dimension of the input tensor.
        eps : float, optional
            A small number to avoid division by zero. Default is 1e-6.
        """
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to normalize.

        Returns
        -------
        torch.Tensor
            The normalized tensor.
        """
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RMSNorm layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class Head(nn.Module):
    """
    A single head in a multi-headed attention layer.
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the Head module.

        Parameters
        ----------
        model_config : ModelConfig
            The configuration object for the model.
        """
        super().__init__()
        self.key = nn.Linear(model_config.n_embd, model_config.head_size, bias=False)
        self.query = nn.Linear(model_config.n_embd, model_config.head_size, bias=False)
        self.value = nn.Linear(model_config.n_embd, model_config.head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(model_config.block_size, model_config.block_size)),
        )

        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Head layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
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
    """
    Multi-headed attention layer with multiple heads in parallel.
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the MultiHeadAttention module.

        Parameters
        ----------
        model_config : ModelConfig
            The configuration object for the model.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(model_config=model_config) for _ in range(model_config.n_head)]
        )
        self.proj = nn.Linear(model_config.n_embd, model_config.n_embd)
        self.dropout = nn.Dropout(model_config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiHeadAttention layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # Concatenate the outputs of each head
        out = self.dropout(
            self.proj(out)
        )  # Apply a linear projection to the concatenated outputs
        return out


class FeedForward(nn.Module):
    """
    Feed-forward layer (two linear layers with a ReLU in between).
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the FeedForward module.

        Parameters
        ----------
        model_config : ModelConfig
            The configuration object for the model.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_config.n_embd, 4 * model_config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * model_config.n_embd, model_config.n_embd),
            nn.Dropout(model_config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeedForward layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.net(x)


class Block(nn.Module):
    """
    A transformer block: Communication followed by computation.
    """

    def __init__(self, model_config: ModelConfig):
        """
        Initialize the Block module.

        Parameters
        ----------
        model_config : ModelConfig
            The configuration object for the model.
        """
        super().__init__()
        model_config.head_size = model_config.n_embd // model_config.n_head
        self.sa = MultiHeadAttention(model_config=model_config)
        self.ffwd = FeedForward(model_config=model_config)
        self.ln1 = RMSNorm(model_config.n_embd, eps=model_config.norm_eps)
        self.ln2 = RMSNorm(model_config.n_embd, eps=model_config.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Block layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NherfLanguageModel(nn.Module):
    """
    Nherf Language Model.
    """

    def __init__(self, model_config: ModelConfig, vocab_size: int):
        """
        Initialize the NherfLanguageModel module.

        Parameters
        ----------
        model_config : ModelConfig
            The configuration object for the model.
        vocab_size : int
            The size of the vocabulary.
        """
        super().__init__()
        self.model_config = model_config
        self.embedding = nn.Embedding(vocab_size, self.model_config.n_embd)
        self.position_embedding = nn.Embedding(
            self.model_config.block_size, self.model_config.n_embd
        )
        self.blocks = nn.Sequential(
            *[
                Block(model_config=self.model_config)
                for _ in range(self.model_config.n_layer)
            ]
        )
        self.ln_f = RMSNorm(
            self.model_config.n_embd, eps=self.model_config.norm_eps
        )  # Final layer norm
        self.lm_head = nn.Linear(
            self.model_config.n_embd, vocab_size
        )  # Final linear layer

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the NherfLanguageModel.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        targets : torch.Tensor, optional
            The target tensor. If not provided, the loss will be None.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the logits and the loss.
        """
        B, T = x.shape
        device = x.device  # Get the device from the input tensor

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

    def generate(self, x: torch.Tensor, max_new_tokens: int = 100) -> torch.Tensor:
        """
        Generate a sequence of tokens.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate. Default is 100.

        Returns
        -------
        torch.Tensor
            The generated sequence of tokens.
        """
        for _ in range(max_new_tokens):
            # crop x to the last model_config.block_size tokens
            x_cond = (
                x
                if x.shape[-1] <= self.model_config.block_size
                else x[:, -self.model_config.block_size :]
            )
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU available, using CUDA")
    else:
        print("GPU not available, using CPU")

    with open("../../data/input.txt", "r") as f:
        text = f.read()

    encode, decode, vocab_size = get_tokenizer(text=text, tokenizer_type="character")

    # --- setup ---

    model = NherfLanguageModel(model_config=ModelConfig, vocab_size=vocab_size)
    model = model.to(device)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = torch.tensor(encode(text), dtype=torch.long)

    # Let's now split up the data into train and validation sets
    n = int(0.9 * len(data))  # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    for iter in range(ModelConfig.max_iters):
        if iter % (ModelConfig.max_iters // 20) == 0:
            print(f"Started training iteration {iter}")
            losses = estimate_loss(
                train_data=train_data,
                val_data=val_data,
                model_config=ModelConfig,
                model=model,
                device=device,
            )
            print(
                f"Step {iter}: Train loss {losses['train']}, Val loss {losses['val']}"
            )

        x_batch, y_batch = get_batch(
            data=train_data,
            model_config=ModelConfig,
            device=device,
        )

        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.tensor(
        encode("NICOLAI:\n"), dtype=torch.long, device=device
    ).unsqueeze(0)
    print("\n=====\n\n")
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
