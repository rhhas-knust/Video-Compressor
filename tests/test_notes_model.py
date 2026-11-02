import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import notes_model as nm

LECTURE = (
    "Today we are going to talk about photosynthesis and how plants make their food. "
    "Photosynthesis happens in the chloroplasts, which contain a green pigment called chlorophyll. "
    "The most important thing to remember is that chlorophyll absorbs sunlight. "
    "Um, so, yeah. "
    "Plants take in carbon dioxide through small openings in their leaves! "
    "Using sunlight, plants turn carbon dioxide and water into glucose and oxygen. "
    "Okay"
)
SUMMARY = "Photosynthesis uses sunlight absorbed by chlorophyll to turn carbon dioxide and water into glucose and oxygen."


def test_split_sentences_matches_page_rules():
    s = nm.split_sentences(LECTURE)
    assert s[0].startswith("Today") and s[0].endswith("food.")
    assert "Um, so, yeah." in s
    assert s[-1] == "Okay"
    assert len(s) == 7
    # like the page, a long stretch with no punctuation stays one sentence
    assert len(nm.split_sentences(" ".join(["word"] * 95))) == 1


def test_model_input_has_sentence_then_neighbours():
    s = ["A one.", "B two.", "C three."]
    assert nm.model_input(s, 1) == "B two. [SEP] A one. C three."
    assert nm.model_input(s, 0) == "A one. [SEP] B two."
    assert nm.model_input(["Only."], 0) == "Only. [SEP] "


def test_oracle_picks_sentences_that_match_the_summary():
    s = nm.split_sentences(LECTURE)
    labels = nm.oracle_labels(s, SUMMARY)
    picked = [x for x, y in zip(s, labels) if y]
    assert any("glucose and oxygen" in x for x in picked)
    assert not any(x.startswith("Um") or x == "Okay" for x in picked)
    assert 1 <= sum(labels) <= 3


def test_rule_scores_and_select():
    s = nm.split_sentences(LECTURE)
    scores = nm.rule_scores(s)
    assert scores[s.index("Um, so, yeah.")] < scores[2]
    picked = nm.select(s, scores, k=3, min_gap_seconds=0)
    assert picked == sorted(picked) and len(picked) == 3


def test_make_examples_balances_labels():
    docs = [{"sentences": nm.split_sentences(LECTURE * 5), "summary": SUMMARY, "source": "t"}]
    rows = nm.make_examples(docs, negatives_per_positive=2)
    pos = sum(r["label"] for r in rows)
    assert pos >= 1 and len(rows) - pos <= 2 * pos + 2
    assert all(" [SEP] " in r["text"] for r in rows)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("onnxruntime") is None,
                    reason="needs torch, transformers and onnxruntime")
def test_train_and_export_tiny_model(tmp_path):
    """Runs the real train() and export_onnx() on a tiny random model built offline."""
    from transformers import ElectraConfig, ElectraForSequenceClassification, ElectraTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", ",", "!", "?"]
    vocab += sorted(set(nm.words(LECTURE + " " + SUMMARY + " thanks everyone any questions let's start deadline")))
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    base = tmp_path / "base"
    tok = ElectraTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    tok.save_pretrained(base)
    cfg = ElectraConfig(vocab_size=len(vocab), embedding_size=16, hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=nm.MAX_LEN)
    ElectraForSequenceClassification(cfg).save_pretrained(base)

    docs = [{"sentences": nm.split_sentences(LECTURE * 4), "summary": SUMMARY, "source": "t"}] * 10
    rows = nm.make_examples(docs)
    work = nm.train(rows, tmp_path / "pt", base_model=str(base), epochs=1, batch_size=8)
    probs = nm.model_scorer(work)(docs[0]["sentences"])
    assert len(probs) == len(docs[0]["sentences"]) and all(0 <= p <= 1 for p in probs)

    out = nm.export_onnx(work, tmp_path / "notes-model")
    assert (out / "onnx" / "model_quantized.onnx").exists()
    assert not (out / "onnx" / "model.onnx").exists()
    assert (out / "tokenizer.json").exists()
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "electra" and cfg["id2label"]["1"] == "important"
