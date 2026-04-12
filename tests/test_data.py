import torch

from mhc_attnres.data import TokenCorpus


def test_token_corpus_batch_shapes():
    corpus = TokenCorpus(tokens=torch.arange(1_000))
    x, y = corpus.get_batch(batch_size=4, context_length=32, device=torch.device("cpu"))
    assert x.shape == (4, 32)
    assert y.shape == (4, 32)
    torch.testing.assert_close(x[:, 1:], y[:, :-1])
