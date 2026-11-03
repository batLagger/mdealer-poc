import matplotlib
matplotlib.use("Agg")

import pytest
import torch

from bridge_generator import calc_losers
from hand_pair_emb import (
    HandLTCfuzzyPair,
    decode_pair,
    encode_pair,
    generate_pair_training_set,
    get_ltc_embedding_fuzzy_pair,
    hands_similarity_fuzzy_pair,
    load_ltc_fuzzy_pair_model,
    pair_labels,
    predict_ltc_pair,
    print_pair2,
    train_ltc_fuzzy_pair,
    verify_pairs,
)


def _hand_from_cards(cards):
    hand = torch.zeros(52)
    hand[list(cards)] = 1
    return hand.reshape(4, 13)


def test_encode_decode_round_trip():
    north = _hand_from_cards(range(0, 13))
    south = _hand_from_cards(range(13, 26))
    pair = encode_pair(north, south)
    assert set(pair.unique().tolist()) == {0.0, 1.0, 2.0}
    n, s = decode_pair(pair)
    assert torch.equal(n, north) and torch.equal(s, south)


def test_encode_rejects_overlap_and_bad_counts():
    north = _hand_from_cards(range(0, 13))
    with pytest.raises(ValueError, match="share"):
        encode_pair(north, _hand_from_cards(range(12, 25)))
    with pytest.raises(ValueError, match="13 cards"):
        encode_pair(north, _hand_from_cards(range(13, 25)))


def test_generated_pairs_are_valid():
    pairs = generate_pair_training_set(200)
    verify_pairs(pairs)
    with pytest.raises(ValueError):
        verify_pairs(pairs * 2)


def test_pair_labels():
    pairs = generate_pair_training_set(20)
    tricks, ltc_n, ltc_s = pair_labels(pairs)
    north, south = decode_pair(pairs)
    for i in range(len(pairs)):
        assert ltc_n[i] == pytest.approx(calc_losers(north[i])[0])
        assert ltc_s[i] == pytest.approx(calc_losers(south[i])[0])
    assert torch.allclose(tricks, 24 - (ltc_n + ltc_s))


def test_forward_shapes_and_embedding_layout():
    model = HandLTCfuzzyPair()
    out = model(generate_pair_training_set(8))
    assert out[0].shape == (8, 1)
    assert out.aux_tricks.shape == out.aux_ltc_north.shape == out.aux_ltc_south.shape == (8, 1)
    for t in (out.aux_shape_north, out.aux_shape_south, out.aux_sorted_shape_north,
              out.aux_sorted_shape_south, out.aux_fit_shape, out.aux_fit_sorted_shape):
        assert t.shape == (8, 4)

    emb = get_ltc_embedding_fuzzy_pair(model, generate_pair_training_set(5))
    assert emb.shape == (5, 64) == (5, model.embedding_size)
    slices = model.embedding_slices()
    assert slices["card"] == slice(0, 16)
    assert slices["fit_sorted_shape"].stop == 64


def test_embedding_is_order_sensitive():
    model = HandLTCfuzzyPair()
    pair = generate_pair_training_set(1)
    swapped = torch.where(pair == 1, 2.0, torch.where(pair == 2, 1.0, 0.0))
    emb = get_ltc_embedding_fuzzy_pair(model, torch.cat([pair, swapped]))
    assert not torch.allclose(emb[0], emb[1])


def test_similarity_ranks_query_first():
    model = HandLTCfuzzyPair()
    pairs = generate_pair_training_set(100)
    idx, scores = hands_similarity_fuzzy_pair(pairs[37], pairs, model)
    assert idx[0].item() == 37
    assert scores[0].item() == pytest.approx(1.0)
    assert torch.all(scores[:-1] >= scores[1:])


def test_print_pair2_layout(capsys):
    north = _hand_from_cards([12, 10, 8, 6, 1] + list(range(13, 21)))  # spades AQT83, hearts 9-2, no diamonds/clubs
    south = _hand_from_cards(list(range(21, 26)) + list(range(26, 34)))
    print_pair2(encode_pair(north, south), gap=10)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "North     South"
    assert lines[1] == "AQT83     ----"
    assert lines[3].startswith("----      ") and lines[4].startswith("----      ")
    assert len(lines) == 7


def test_train_save_load_predict(tmp_path):
    path = tmp_path / "pair.pth"
    model = train_ltc_fuzzy_pair(num_pairs=512, epochs=1, batch_size=64, plot=False,
                                 save_path=str(path), device="cpu")
    loaded = load_ltc_fuzzy_pair_model(str(path), device="cpu")
    pairs = generate_pair_training_set(4)
    assert torch.allclose(predict_ltc_pair(model, pairs), predict_ltc_pair(loaded, pairs))
    assert predict_ltc_pair(loaded, pairs[0]).ndim == 0
