vocab_size = 65


class SmallNherformerConfig:
    block_size = 16
    batch_size = 8
    max_iters = 100
    eval_iters = 300
    n_embd = 64
    n_head = 4
    n_layer = 4
    dropout = 0.2
    norm_eps = 1e-5


class MediumNherformerConfig:
    block_size = 64
    batch_size = 128 * 2
    max_iters = 2500
    eval_iters = 300
    n_embd = 64 * 6
    n_head = 8
    n_layer = 8
    dropout = 0.2
    norm_eps = 1e-5
