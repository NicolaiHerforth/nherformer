from pathlib import Path
import modal

from nherformer.prod.transformer_medium import estimate_loss, get_batch


stub = modal.Stub(name="trainsformer")


current_dir = Path(__file__).resolve().parent
pyproject_path = current_dir.parent.parent / "pyproject.toml"

image = (
    modal.Image.from_registry("python:3.11-slim-buster")
    .poetry_install_from_file(str(pyproject_path))
    .copy_local_file(pyproject_path, "/root/pyproject.toml")
    .apt_install("wget")
    .run_commands(
        "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -P /root/"
    )
)

if not modal.is_local():
    pyproject_path = Path("/root/pyproject.toml")


@stub.function(image=image, gpu="A100", timeout=60 * 100)
@modal.web_endpoint(method="GET", label="trainsformer")
def trainsformer():
    from nherformer.prod.transformer_medium import (
        NherfLanguageModel,
        estimate_loss,
        get_batch,
    )

    import torch
    from nherformer.prod.config import MediumNherformerConfig as ModelConfig
    from nherformer.prod.utils.helper_funcs import get_tokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU available, using CUDA")
    else:
        print("GPU not available, using CPU")

    with open("/root/input.txt", "r") as f:
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
