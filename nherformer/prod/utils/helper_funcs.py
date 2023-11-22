from typing import Literal
import torch
import torch.nn as nn
from nherformer.prod.config import MediumNherformerConfig as ModelConfig


def get_batch(
    data: torch.Tensor, model_config: ModelConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a small batch of data of inputs x and targets y.

    Parameters
    ----------
    data : torch.Tensor
        The data tensor.
    model_config : ModelConfig
        The configuration object for the model.
    device : torch.device
        The device to which the tensors will be moved.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the input and target tensors.
    """
    ix = torch.randint(len(data) - model_config.block_size, (model_config.batch_size,))
    x = torch.stack([data[i : i + model_config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + model_config.block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    model_config: ModelConfig,
    model: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Estimate the loss for the given model configuration and model.

    Parameters
    ----------
    train_data : torch.Tensor
        The training data tensor.
    val_data : torch.Tensor
        The validation data tensor.
    model_config : ModelConfig
        The configuration object for the model.
    model : nn.Module
        The model to estimate the loss for.
    device : torch.device
        The device to which the tensors will be moved.

    Returns
    -------
    dict[str, float]
        A dictionary containing the mean loss for the 'train' and 'val' splits.
    """
    out = {}
    model.eval()
    for split, data in zip(["train", "val"], [train_data, val_data]):
        losses = torch.zeros(model_config.eval_iters)
        for k in range(model_config.eval_iters):
            X, Y = get_batch(data=data, model_config=model_config, device=device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_tokenizer(text: str, tokenizer_type: Literal["character", "subword"]):
    """
    Get the tokenizer based on the tokenizer type.

    Parameters
    ----------
    text : str
        The text to be tokenized.
    tokenizer_type : Literal["character", "subword"]
        The type of tokenizer. Can be either 'character' or 'subword'.

    Returns
    -------
    tuple
        A tuple containing the encode function, decode function, and vocab size.
    """
    if tokenizer_type not in ["character", "subword"]:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")
    # Overwriting the vocab size to be the number of unique characters
    if tokenizer_type == "character":
        print("Setting up character tokenizer")
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        stoi = {s: i for i, s in enumerate(set(chars))}
        itos = {i: s for i, s in enumerate(set(chars))}
        encode = lambda s: [
            stoi[c] for c in s
        ]  # encoder: string -> list of indices (tokens)
        decode = lambda s: "".join(
            [itos[i] for i in s]
        )  # decoder: list of indices (tokens) -> string

    else:
        print("Setting up subword tokenizer with tiktoken")
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt2")
        assert enc.decode(enc.encode(text)) == text

        vocab_size = enc.n_vocab

        encode = lambda s: enc.encode(s)
        decode = lambda s: enc.decode(s)
    return encode, decode, vocab_size
