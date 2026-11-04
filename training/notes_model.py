"""Train the "important points" model used by the web app's Transcribe & notes tool.

The model reads one transcript sentence plus the sentences around it and predicts
whether it belongs in the notes. It is a small ELECTRA classifier, exported to ONNX
and quantized to int8 (about 14 MB) so transformers.js can run it in the browser.

Labels come from public datasets that pair long texts with human summaries: a
sentence counts as important when it is part of the greedy ROUGE "oracle", the set
of sentences that best reproduces the summary (the usual way to train extractive
summarizers). Meeting transcripts (MeetingBank) are closest to spoken lectures;
news articles (CNN/DailyMail) add volume.

Run it from the Colab notebook in this folder, or:  python notes_model.py --help
"""
import argparse
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path

BASE_MODEL = "google/electra-small-discriminator"
MAX_LEN = 128
LABELS = {0: "other", 1: "important"}

# ---------------------------------------------------------------- text handling
# These mirror pickNotes() in web/index.html, so training sees sentences the same
# way the page builds them from Whisper's output.

SENTENCE_RE = re.compile(r"""[^.!?]+[.!?]+["'”’)\]]*|[^.!?]+$""")
WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
MAX_SENTENCE_WORDS = 40

STOPWORDS = set(
    "a about above after again against all also am an and any are as at be because been before being below "
    "between both but by can could did do does doing down during each even few for from further get got had "
    "has have having he her here hers him his how i if in into is it its just know like make many me might "
    "more most much my no nor not now of off on once one only or other our out over own really right said "
    "same say see she should so some such than that the their them then there these they thing things think "
    "this those through to too um uh under until up very was way we well were what when where which while "
    "who why will with would yeah yes you your going gonna want okay ok oh lot let us".split()
)
CUES = re.compile(
    r"\b(important|importantly|remember|key|note|notice|main|summary|summari[sz]e|conclusion|in short|must|"
    r"need to|don'?t forget|the point|crucial|essential|critical|takeaway|lesson|first|second|third|finally|"
    r"always|never)\b",
    re.I,
)


def words(text):
    return WORD_RE.findall(text.lower())


def split_sentences(text):
    """Splits text into sentences, also breaking run-ons longer than 40 words."""
    out, cur = [], ""
    for piece in SENTENCE_RE.findall(text.replace("\n", " ")):
        piece = piece.strip()
        if not piece:
            continue
        cur = (cur + " " + piece).strip()
        if re.search(r"""[.!?]["'”’)\]]*$""", piece) or len(words(cur)) > MAX_SENTENCE_WORDS:
            out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def model_input(sentences, i):
    """The text the classifier sees: the sentence, then its neighbours as context.

    web/index.html builds exactly this string; keep the two in step.
    """
    context = " ".join(s for s in (sentences[i - 1] if i > 0 else "", sentences[i + 1] if i + 1 < len(sentences) else "") if s)
    return sentences[i] + " [SEP] " + context


# ---------------------------------------------------------------- labels


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap, cand, ref):
    if not cand or not ref or not overlap:
        return 0.0
    p, r = overlap / cand, overlap / ref
    return 2 * p * r / (p + r)


def oracle_labels(sentences, summary, max_selected=None):
    """Greedy ROUGE-1+2 oracle: adds the sentence that most improves overlap with the
    summary until nothing improves it. Returns a 0/1 label per sentence."""
    ref_tokens = words(summary)
    ref1, ref2 = _ngrams(ref_tokens, 1), _ngrams(ref_tokens, 2)
    n_ref1, n_ref2 = sum(ref1.values()), sum(ref2.values())
    toks = [words(s) for s in sentences]
    uni = [_ngrams(t, 1) for t in toks]
    bi = [_ngrams(t, 2) for t in toks]
    if max_selected is None:
        max_selected = max(3, len(split_sentences(summary)))
    chosen, cur1, cur2, best = [], Counter(), Counter(), 0.0
    o1 = o2 = len1 = len2 = 0  # clipped overlaps with the summary, and sizes, of the chosen set
    # Only sentences sharing a word with the summary can help; this keeps long transcripts fast.
    candidates = [i for i in range(len(sentences)) if any(w in ref1 for w in uni[i])]

    def gain(cur, add, ref):
        # how much the clipped overlap grows if `add` joins `cur`
        return sum(min(cur[g] + c, ref[g]) - min(cur[g], ref[g]) for g, c in add.items() if g in ref)

    while len(chosen) < max_selected:
        pick, pick_score, pick_o = None, best, None
        for i in candidates:
            if i in chosen:
                continue
            n1, n2 = o1 + gain(cur1, uni[i], ref1), o2 + gain(cur2, bi[i], ref2)
            score = _f1(n1, len1 + len(toks[i]), n_ref1) + _f1(n2, len2 + max(0, len(toks[i]) - 1), n_ref2)
            if score > pick_score:
                pick, pick_score, pick_o = i, score, (n1, n2)
        if pick is None:
            break
        chosen.append(pick)
        cur1.update(uni[pick]); cur2.update(bi[pick])
        o1, o2 = pick_o
        len1 += len(toks[pick]); len2 += max(0, len(toks[pick]) - 1)
        best = pick_score
    labels = [0] * len(sentences)
    for i in chosen:
        labels[i] = 1
    return labels


# ---------------------------------------------------------------- the page's current rules


def rule_scores(sentences):
    """Python port of the page's rule-based scoring, used as the baseline to beat."""
    freq = Counter()
    for s in sentences:
        for w in words(s):
            if w not in STOPWORDS and len(w) >= 3 and not w.isdigit():
                freq[w] += 1
    max_f = max([1] + list(freq.values()))
    scores = []
    for s in sentences:
        ws = words(s)
        content = [w for w in ws if w in freq]
        score = sum(math.log(1 + freq[w]) / math.log(1 + max_f) for w in set(content))
        score = score / math.sqrt(len(content)) if content else 0.0
        if CUES.search(s):
            score *= 1.5
        if len(ws) < 6:
            score *= 0.3
        if len(ws) > 45:
            score *= 0.7
        scores.append(score)
    return scores


def select(sentences, scores, k, min_gap_seconds=15.0, words_per_second=2.5):
    """Same choice rule as the page: best scores first, at least 15 s apart, in time order."""
    starts, total = [], 0
    for s in sentences:
        starts.append(total / words_per_second)
        total += len(words(s))
    chosen = []
    for i in sorted(range(len(sentences)), key=lambda i: -scores[i]):
        if len(chosen) >= k or scores[i] <= 0:
            break
        if all(abs(starts[i] - starts[j]) >= min_gap_seconds for j in chosen):
            chosen.append(i)
    return sorted(chosen)


# ---------------------------------------------------------------- datasets


def load_documents(max_meetings=1200, max_news=6000, seed=0):
    """Returns {"train": [...], "test": [...]}, each a list of {"sentences", "summary", "source"}.

    Each dataset is optional: if one can't be downloaded the others are still used.
    """
    from datasets import load_dataset

    random.seed(seed)
    docs = {"train": [], "test": []}
    sources = [
        # (name, loader, text field, summary field, max docs)
        ("meetingbank", lambda split: load_dataset("huuuyeah/meetingbank", split=split), "transcript", "summary", max_meetings),
        ("cnn_dailymail", lambda split: load_dataset("abisee/cnn_dailymail", "3.0.0", split=split), "article", "highlights", max_news),
    ]
    for name, loader, text_field, summary_field, cap in sources:
        for split, hf_split, share in (("train", "train", 1.0), ("test", "test", 0.1)):
            try:
                ds = loader(hf_split)
            except Exception as e:  # network, renamed dataset, etc.
                print(f"  skipped {name}/{hf_split}: {e}")
                continue
            n = min(len(ds), max(20, int(cap * share)))
            ds = ds.shuffle(seed=seed).select(range(n))
            for row in ds:
                sentences = split_sentences(row[text_field] or "")
                summary = (row[summary_field] or "").replace("\n", " ")
                if len(sentences) >= 5 and summary.strip():
                    docs[split].append({"sentences": sentences, "summary": summary, "source": name})
            print(f"  {name}/{hf_split}: {n} documents")
    if not docs["train"]:
        raise RuntimeError("No training data could be downloaded. Check the Colab internet connection.")
    return docs


def make_examples(docs, negatives_per_positive=4, max_sentences=800, seed=0):
    """Labels every document and turns it into (text, label) rows, keeping a few
    negatives per positive so the classifier isn't swamped by unimportant sentences."""
    rng = random.Random(seed)
    rows = []
    for d in docs:
        sentences = d["sentences"][:max_sentences]
        labels = oracle_labels(sentences, d["summary"])
        d["labels"] = labels
        pos = [i for i, y in enumerate(labels) if y]
        neg = [i for i, y in enumerate(labels) if not y]
        rng.shuffle(neg)
        for i in pos + neg[: max(2, len(pos) * negatives_per_positive)]:
            rows.append({"text": model_input(sentences, i), "label": labels[i]})
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------- training


def train(rows, out_dir, base_model=BASE_MODEL, epochs=2, batch_size=64, lr=1e-4):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=LABELS, label2id={v: k for k, v in LABELS.items()})
    split = int(len(rows) * 0.95)
    ds = {
        "train": Dataset.from_list(rows[:split]),
        "eval": Dataset.from_list(rows[split:]),
    }
    ds = {k: v.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True) for k, v in ds.items()}

    def metrics(p):
        pred = np.argmax(p.predictions, axis=-1)
        tp = int(((pred == 1) & (p.label_ids == 1)).sum())
        prec = tp / max(1, int((pred == 1).sum()))
        rec = tp / max(1, int((p.label_ids == 1).sum()))
        return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec)}

    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        warmup_steps=max(1, int(0.06 * epochs * math.ceil(len(ds["train"]) / batch_size))),
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["eval"],
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    print(trainer.evaluate())
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def model_scorer(model_dir, batch_size=64):
    """Returns a function giving the probability that each sentence is important,
    using a saved (PyTorch) model. Loads the model once, on the GPU when there is one."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()

    def score(sentences):
        texts = [model_input(sentences, i) for i in range(len(sentences))]
        probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = tok(texts[i:i + batch_size], truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
                probs += torch.softmax(model(**enc).logits, -1)[:, 1].tolist()
        return probs

    return score


# ---------------------------------------------------------------- evaluation


def evaluate(test_docs, score_fns, k_max=20):
    """ROUGE of each method's chosen sentences against the human summary, averaged per source."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    results = {}
    for d in test_docs:
        sentences = d["sentences"][:800]
        k = max(3, min(k_max, len(split_sentences(d["summary"]))))
        for name, fn in score_fns.items():
            picked = select(sentences, fn(sentences), k)
            s = scorer.score(d["summary"], " ".join(sentences[i] for i in picked))
            bucket = results.setdefault((d["source"], name), {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "n": 0})
            for m in ("rouge1", "rouge2", "rougeL"):
                bucket[m] += s[m].fmeasure
            bucket["n"] += 1
    table = []
    for (source, name), b in sorted(results.items()):
        n = b.pop("n")
        table.append({"source": source, "method": name, **{m: round(100 * v / n, 2) for m, v in b.items()}, "docs": n})
    return table


# ---------------------------------------------------------------- export for the browser


def export_onnx(model_dir, out_dir):
    """Writes the layout transformers.js loads: config + tokenizer files and
    onnx/model_quantized.onnx (int8). Checks the ONNX output against PyTorch."""
    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    model.config.return_dict = True

    sample = ["The most important thing to remember is the deadline. [SEP] Thanks everyone. Any questions?",
              "Okay. [SEP] Let's start."]
    enc = tok(sample, padding=True, return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc]

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, *args):
            return self.m(**dict(zip(names, args))).logits

    fp32 = out / "onnx" / "model.onnx"
    # The wrapper must be in eval mode too: the exporter restores the wrapper's mode afterwards,
    # which would otherwise switch the model back to training (dropout on) for the checks below.
    torch.onnx.export(
        Wrapper(model).eval(), tuple(enc[n] for n in names), str(fp32),
        input_names=names, output_names=["logits"],
        dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32), str(out / "onnx" / "model_quantized.onnx"), weight_type=QuantType.QInt8)
    model.eval()

    # Check against PyTorch on batch sizes and lengths other than the traced sample, as the page sends.
    checks = [sample + ["Short one. [SEP] ", "A much longer sentence about the budget, the schedule and who does what next week. [SEP] Okay."],
              ["Just one sentence here. [SEP] Then the next one."]]
    # The float model should match almost exactly; int8 rounding moves probabilities a little.
    for f, tolerance in (("model.onnx", 1e-3), ("model_quantized.onnx", 0.05)):
        sess = ort.InferenceSession(str(out / "onnx" / f), providers=["CPUExecutionProvider"])
        worst = 0.0
        for batch in checks:
            e = tok(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                ref = torch.softmax(model(**e).logits, -1).numpy()
            logits = sess.run(["logits"], {n: e[n].numpy() for n in names})[0]
            probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
            worst = max(worst, float(np.abs(probs - ref).max()))
        print(f"  {f}: max difference from PyTorch {worst:.4f}")
        if worst > tolerance:
            raise RuntimeError(f"{f} disagrees with PyTorch by {worst:.3f}; the export is not usable")

    fp32.unlink()  # the page only loads the int8 model
    model.config.save_pretrained(out)
    tok.save_pretrained(out)
    size = (out / "onnx" / "model_quantized.onnx").stat().st_size
    print(f"  wrote {out} ({size / 1e6:.1f} MB model)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="notes-model", help="folder for the browser-ready model")
    ap.add_argument("--work", default="notes-model-pt", help="folder for the PyTorch checkpoint")
    ap.add_argument("--meetings", type=int, default=1200, help="MeetingBank documents to use")
    ap.add_argument("--news", type=int, default=6000, help="CNN/DailyMail articles to use")
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)

    print("Downloading datasets…")
    docs = load_documents(args.meetings, args.news)
    print("Labelling sentences…")
    rows = make_examples(docs["train"])
    print(f"  {len(rows)} training sentences, {sum(r['label'] for r in rows)} important")
    print("Training…")
    train(rows, args.work, epochs=args.epochs)
    print("Comparing with the current rules on held-out documents…")
    table = evaluate(docs["test"], {
        "trained model": model_scorer(args.work),
        "current rules": rule_scores,
        "first sentences": lambda s: [1.0 / (i + 1) for i in range(len(s))],
    })
    for row in table:
        print("  ", json.dumps(row))
    print("Exporting for the browser…")
    export_onnx(args.work, args.out)


if __name__ == "__main__":
    main()
