"""Train the on-device Twi (Akan) models used by the web app's Transcribe & notes tool.

Two small models, each exported to int8 ONNX so transformers.js can run them in the browser:

* Speech recognition: Whisper base fine-tuned on public Twi/Akan speech. Whisper has no Akan
  language token, so the model is trained on Yoruba's token slot (LANGUAGE_TOKEN) and the page
  asks for the same one. About 80 MB in the browser.
* Translation: Helsinki-NLP's many-to-English Marian model fine-tuned on Twi-English sentence
  pairs, used to add an English line under each Twi line. About 80 MB.

Every dataset is optional: the ones that can't be downloaded (renamed, gated, offline) are
skipped. Run it from the Colab notebook in this folder.

Package versions are pinned in requirements-twi.txt to the set transformers.js is tested with.
"""
import json
import random
import re
import shutil
import unicodedata
from pathlib import Path

BASE_ASR = "openai/whisper-base"
BASE_MT = "Helsinki-NLP/opus-mt-mul-en"
LANGUAGE_TOKEN = "yoruba"   # web/index.html passes the same language to Whisper
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30

# (dataset id, config). Public Twi/Akan speech with transcripts; each one is tried and skipped if
# it can't be loaded. Gated sets need `huggingface_hub.login()` with an account that accepted them.
ASR_DATASETS = [
    ("ghananlpcommunity/akuapem_multispeaker_audio_transcribed", None),
    ("michsethowusu/twi_multispeaker_audio_transcribed", None),
    ("ghananlpcommunity/twi-speech-text-multispeaker-16k", None),
    ("Kennethdot/Ghana_English-Twi_Code-switching_ASR", None),
]
MT_DATASETS = [
    ("Ghana-NLP/TWI_ENGLISH_PARALLEL_TEXT", None),
]
TEXT_COLUMNS = ("transcription", "text", "sentence", "transcript", "normalized_text", "Transcription", "Text", "label")


# ---------------------------------------------------------------- text


def clean_text(text):
    """Tidies a transcript for training: Unicode NFC (keeps ɛ and ɔ intact), straight quotes, single spaces."""
    text = unicodedata.normalize("NFC", str(text or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_wer(text):
    """Lowercase, no punctuation: how word error rate is usually scored."""
    text = clean_text(text).lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_text_column(column_names):
    for c in TEXT_COLUMNS:
        if c in column_names:
            return c
    return None


def find_pair_columns(row):
    """Finds (twi, english) fields in a parallel-text row, whatever the dataset calls them."""
    if isinstance(row.get("translation"), dict):
        row = row["translation"]
    keys = list(row.keys())
    twi = next((k for k in keys if re.fullmatch(r"(?i)(twi|tw|ak|aka|akan|asante.*|source|src)", k)), None)
    eng = next((k for k in keys if re.fullmatch(r"(?i)(english|en|eng|target|tgt)", k)), None)
    return (row, twi, eng) if twi and eng else (row, None, None)


# ---------------------------------------------------------------- speech data


def load_asr_data(max_hours=40, test_size=300, seed=0, datasets_list=ASR_DATASETS):
    """Returns (train, test) datasets with columns `audio` (16 kHz) and `text`."""
    from datasets import Audio, concatenate_datasets, load_dataset

    parts = []
    for name, config in datasets_list:
        try:
            ds = load_dataset(name, config)
        except Exception as e:
            print(f"  skipped {name}: {str(e).splitlines()[0][:160]}")
            continue
        splits = [ds[s] for s in ds.keys()]
        ds = concatenate_datasets(splits) if len(splits) > 1 else splits[0]
        audio_col = "audio" if "audio" in ds.column_names else next((c for c in ds.column_names if ds.features[c].__class__.__name__ == "Audio"), None)
        text_col = find_text_column(ds.column_names)
        if not audio_col or not text_col:
            print(f"  skipped {name}: no audio/transcript columns in {ds.column_names}")
            continue
        ds = ds.select_columns([audio_col, text_col]).rename_columns({audio_col: "audio", text_col: "text"})
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
        ds = ds.map(lambda r: {"text": clean_text(r["text"])}).filter(lambda r: len(r["text"]) > 1)
        print(f"  {name}: {len(ds)} clips")
        parts.append(ds)
    if not parts:
        raise RuntimeError("No Twi speech data could be loaded. Check the internet connection (and login for gated sets).")
    data = concatenate_datasets(parts).shuffle(seed=seed)
    data = data.map(lambda r: {"seconds": len(r["audio"]["array"]) / SAMPLE_RATE}, desc="measuring clips")
    data = data.filter(lambda r: 0.5 <= r["seconds"] <= MAX_AUDIO_SECONDS)
    split = data.train_test_split(test_size=min(test_size, max(1, len(data) // 10)), seed=seed)
    train, test = split["train"], split["test"]
    # keep training time predictable: cap the hours of audio
    total, keep = 0.0, []
    for i, s in enumerate(train["seconds"]):
        if total + s > max_hours * 3600:
            break
        total += s
        keep.append(i)
    train = train.select(keep)
    print(f"  training on {len(train)} clips ({total / 3600:.1f} h), testing on {len(test)}")
    return train, test


class SpeechCollator:
    """Turns raw clips into Whisper inputs per batch, so the whole dataset never sits in memory as features."""

    def __init__(self, processor, decoder_start_token_id):
        self.processor = processor
        self.start = decoder_start_token_id

    def __call__(self, rows):
        import torch

        audio = [r["audio"]["array"] for r in rows]
        feats = self.processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        labels = self.processor.tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=440, return_tensors="pt")
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.start).all():
            ids = ids[:, 1:]   # the model adds the start token itself
        return {"input_features": feats.to(torch.float32), "labels": ids}


def _device_dtype(model):
    """Device and dtype of a PyTorch model; CPU float32 for an ONNX Runtime model, which has no parameters."""
    import torch

    try:
        p = next(model.parameters())
        return p.device, p.dtype
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu"), torch.float32


def _is_torch(model):
    import torch

    return isinstance(model, torch.nn.Module) and "_modules" in vars(model)


def asr_processor(base=BASE_ASR):
    from transformers import WhisperProcessor

    processor = WhisperProcessor.from_pretrained(base)
    processor.tokenizer.set_prefix_tokens(language=LANGUAGE_TOKEN, task="transcribe")
    if processor.tokenizer.pad_token is None:   # Whisper pads with <|endoftext|>
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def transcribe(model, processor, clips, batch_size=16):
    """Transcribes clips (lists of 16 kHz samples) with the given model, using the Twi language slot."""
    import torch

    device, dtype = _device_dtype(model)
    out = []
    if _is_torch(model):
        model.eval()
    for i in range(0, len(clips), batch_size):
        feats = processor.feature_extractor(clips[i:i + batch_size], sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats.to(device, dtype=dtype), language=LANGUAGE_TOKEN, task="transcribe", max_new_tokens=225)
        out += processor.tokenizer.batch_decode(ids, skip_special_tokens=True)
    return [clean_text(t) for t in out]


def word_error_rate(references, hypotheses):
    import jiwer

    refs = [normalize_for_wer(r) for r in references]
    hyps = [normalize_for_wer(h) for h in hypotheses]
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def train_asr(train, test, out_dir, base=BASE_ASR, max_steps=3000, batch_size=16, lr=1e-5, eval_samples=200):
    """Fine-tunes Whisper on the Twi clips and reports word error rate before and after."""
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration

    processor = asr_processor(base)
    model = WhisperForConditionalGeneration.from_pretrained(base)
    model.generation_config.language = LANGUAGE_TOKEN
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample = test.select(range(min(eval_samples, len(test))))
    refs, clips = sample["text"], [a["array"] for a in sample["audio"]]
    before = word_error_rate(refs, transcribe(model.to(device), processor, clips))
    print(f"  word error rate before training: {before:.1%}")

    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(out_dir) / "checkpoints"),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=min(500, max_steps // 10),
        max_steps=max_steps,
        gradient_checkpointing=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=2 if torch.cuda.is_available() else 0,
        report_to=[],
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train,
                             data_collator=SpeechCollator(processor, model.config.decoder_start_token_id))
    trainer.train()
    model.config.use_cache = True

    hyps = transcribe(model, processor, clips)
    after = word_error_rate(refs, hyps)
    print(f"  word error rate after training:  {after:.1%}")
    for r, h in list(zip(refs, hyps))[:5]:
        print(f"    said:  {r}\n    heard: {h}")
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)
    return {"wer_before": before, "wer_after": after}


# ---------------------------------------------------------------- translation data


def load_mt_data(test_size=500, seed=0, datasets_list=MT_DATASETS, max_pairs=200000):
    """Returns (train, test) lists of {"twi", "en"} sentence pairs."""
    from datasets import load_dataset

    pairs = []
    for name, config in datasets_list:
        try:
            ds = load_dataset(name, config)
        except Exception as e:
            print(f"  skipped {name}: {str(e).splitlines()[0][:160]}")
            continue
        n = 0
        for split in ds.keys():
            for row in ds[split]:
                r, twi, eng = find_pair_columns(row)
                if not twi:
                    print(f"  skipped {name}: can't tell the Twi and English columns apart in {list(row.keys())}")
                    break
                a, b = clean_text(r[twi]), clean_text(r[eng])
                if a and b and len(a) < 400 and len(b) < 400:
                    pairs.append({"twi": a, "en": b}); n += 1
        print(f"  {name}: {n} sentence pairs")
    if not pairs:
        raise RuntimeError("No Twi-English sentence pairs could be loaded.")
    random.Random(seed).shuffle(pairs)
    pairs = pairs[:max_pairs]
    return pairs[test_size:], pairs[:test_size]


def translate(model, tokenizer, texts, batch_size=32):
    import torch

    device, _ = _device_dtype(model)
    out = []
    if _is_torch(model):
        model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=128, num_beams=4)
        out += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return out


def bleu(references, hypotheses):
    import sacrebleu

    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def train_mt(train, test, out_dir, base=BASE_MT, epochs=3, batch_size=32, lr=5e-5, max_steps=-1):
    """Fine-tunes the Marian translator on Twi-English pairs and reports BLEU before and after."""
    import torch
    from datasets import Dataset
    from transformers import (DataCollatorForSeq2Seq, MarianMTModel, MarianTokenizer, Seq2SeqTrainer,
                              Seq2SeqTrainingArguments)

    tok = MarianTokenizer.from_pretrained(base)
    model = MarianMTModel.from_pretrained(base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    src, refs = [p["twi"] for p in test], [p["en"] for p in test]
    before = bleu(refs, translate(model.to(device), tok, src))
    print(f"  BLEU before training: {before:.1f}")

    ds = Dataset.from_list(train).map(
        lambda b: tok(b["twi"], text_target=b["en"], truncation=True, max_length=128),
        batched=True, remove_columns=["twi", "en"])
    args = Seq2SeqTrainingArguments(
        output_dir=str(Path(out_dir) / "checkpoints"),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        warmup_steps=200,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
    )
    Seq2SeqTrainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForSeq2Seq(tok, model=model)).train()

    hyps = translate(model, tok, src)
    after = bleu(refs, hyps)
    print(f"  BLEU after training:  {after:.1f}")
    for s, r, h in list(zip(src, refs, hyps))[:5]:
        print(f"    Twi:        {s}\n    reference:  {r}\n    translated: {h}")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return {"bleu_before": before, "bleu_after": after}


# ---------------------------------------------------------------- export for the browser
# Follows transformers.js's own converter (scripts/convert.py and scripts/quantize.py, Apache-2.0):
# optimum's ONNX export with past key values, then 8-bit dynamic quantization that uses unsigned
# weights for models with Conv layers (onnxruntime-web has no signed ConvInteger) and quantizes
# inside the merged decoder's subgraphs.


def _quantize(src, dst):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.registry import IntegerOpsRegistry

    ops = {n.op_type for n in onnx.load(str(src), load_external_data=False).graph.node}
    weight = QuantType.QUInt8 if "Conv" in ops else QuantType.QInt8
    quantize_dynamic(str(src), str(dst), weight_type=weight, per_channel=False, reduce_range=False,
                     op_types_to_quantize=list(IntegerOpsRegistry.keys()),
                     extra_options={"EnableSubgraph": True, "MatMulConstBOnly": True})


def _export(model_dir, out_dir, task, files):
    import functools
    import inspect

    import torch
    from optimum.exporters.onnx import main_export

    out = Path(out_dir)
    tmp = out.parent / (out.name + "-fp32")
    for d in (out, tmp):
        if d.exists():
            shutil.rmtree(d)
    # PyTorch 2.9+ defaults torch.onnx.export to its new exporter, which optimum's export code
    # predates; use the classic TorchScript exporter it was written for while exporting.
    original = torch.onnx.export
    if "dynamo" in inspect.signature(original).parameters:
        torch.onnx.export = functools.partial(original, dynamo=False)
    try:
        main_export(str(model_dir), output=str(tmp), task=task, do_validation=False)
    finally:
        torch.onnx.export = original
    (out / "onnx").mkdir(parents=True)
    for f in files:
        _quantize(tmp / f"{f}.onnx", out / "onnx" / f"{f}_quantized.onnx")
    for p in tmp.iterdir():
        if p.suffix != ".onnx" and p.is_file() and not p.name.endswith(".onnx_data"):
            shutil.copy(p, out / p.name)
    return out, tmp


def export_asr(model_dir, out_dir, check_audio=None):
    """Writes the Whisper model as transformers.js loads it: onnx/encoder_model_quantized.onnx and
    onnx/decoder_model_merged_quantized.onnx plus config, generation config, tokenizer and
    feature extractor files. With `check_audio` (a list of 16 kHz clips), checks the int8 model
    writes the same text as PyTorch."""
    out, tmp = _export(model_dir, out_dir, "automatic-speech-recognition-with-past", ["encoder_model", "decoder_model_merged"])
    if check_audio:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained(model_dir)
        ref = transcribe(WhisperForConditionalGeneration.from_pretrained(model_dir), processor, check_audio)
        for label, folder, sub, suffix in (("float32", tmp, "", ""), ("int8", out, "onnx", "_quantized")):
            ort = ORTModelForSpeechSeq2Seq.from_pretrained(folder, subfolder=sub, encoder_file_name=f"encoder_model{suffix}.onnx",
                                                           decoder_file_name=f"decoder_model_merged{suffix}.onnx", use_merged=True)
            got = transcribe(ort, processor, check_audio)
            same = sum(a == b for a, b in zip(ref, got))
            print(f"  {label} ONNX writes the same text as PyTorch on {same} of {len(ref)} test clips")
            if label == "float32" and same < len(ref):
                raise RuntimeError("the exported model does not match PyTorch; the export is not usable")
    shutil.rmtree(tmp)
    _report(out)
    return out


def marian_tokenizer_json(model_dir, tokenizer):
    """One tokenizer.json for Marian's separate source/target SentencePiece models
    (as transformers.js's scripts/extra/marian.py builds it)."""
    vocab = json.loads((Path(model_dir) / "vocab.json").read_text(encoding="utf-8"))
    special = tokenizer.all_special_tokens
    src, tgt = tokenizer.spm_source, tokenizer.spm_target
    src_scores = {src.IdToPiece(i): src.GetScore(i) for i in range(src.GetPieceSize())}
    tgt_scores = {tgt.IdToPiece(i): tgt.GetScore(i) for i in range(tgt.GetPieceSize())}
    eos = tokenizer.eos_token
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [dict(id=vocab.get(x), special=True, content=x, single_word=False, lstrip=False, rstrip=False, normalized=False) for x in special],
        "normalizer": {"type": "Precompiled", "precompiled_charsmap": None},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True}]},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": eos, "type_id": 0}}],
            "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": eos, "type_id": 0}},
                     {"Sequence": {"id": "B", "type_id": 0}}, {"SpecialToken": {"id": eos, "type_id": 0}}],
            "special_tokens": {eos: {"id": eos, "ids": [tokenizer.eos_token_id], "tokens": [eos]}},
        },
        "decoder": {"type": "Metaspace", "replacement": "▁", "add_prefix_space": True},
        "model": {"type": "Unigram", "unk_id": 2, "vocab": [
            [k, 0.0 if k in special else max(src_scores.get(k, -100), tgt_scores.get(k, -100))] for k in vocab]},
    }


def export_mt(model_dir, out_dir, check_texts=None):
    """Writes the translator as transformers.js loads it, with a merged tokenizer.json. With
    `check_texts`, checks the int8 model translates them like PyTorch."""
    from transformers import MarianTokenizer

    out, tmp = _export(model_dir, out_dir, "text2text-generation-with-past", ["encoder_model", "decoder_model_merged"])
    tok = MarianTokenizer.from_pretrained(model_dir)
    (out / "tokenizer.json").write_text(json.dumps(marian_tokenizer_json(model_dir, tok), ensure_ascii=False), encoding="utf-8")
    if check_texts:
        import torch
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import MarianMTModel

        pt = MarianMTModel.from_pretrained(model_dir).eval()
        ref = translate(pt, tok, check_texts)
        enc = tok(check_texts, return_tensors="pt", padding=True)
        start = torch.full((len(check_texts), 1), pt.config.decoder_start_token_id)
        with torch.no_grad():
            ref_logits = pt(**enc, decoder_input_ids=start).logits
        for label, folder, sub, suffix in (("float32", tmp, "", ""), ("int8", out, "onnx", "_quantized")):
            ort = ORTModelForSeq2SeqLM.from_pretrained(folder, subfolder=sub, encoder_file_name=f"encoder_model{suffix}.onnx",
                                                       decoder_file_name=f"decoder_model_merged{suffix}.onnx", use_merged=True)
            with torch.no_grad():
                diff = float((ort(**enc, decoder_input_ids=start).logits - ref_logits).abs().max())
            same = sum(a == b for a, b in zip(ref, translate(ort, tok, check_texts)))
            print(f"  {label} ONNX: largest score difference from PyTorch {diff:.2g}; translates {same} of {len(ref)} sentences identically")
            if label == "float32" and diff > 1e-3:
                raise RuntimeError("the exported translator does not match PyTorch; the export is not usable")
    shutil.rmtree(tmp)
    _report(out)
    return out


def _report(out):
    size = sum(p.stat().st_size for p in Path(out).rglob("*") if p.is_file())
    print(f"  wrote {out} ({size / 1e6:.0f} MB): {', '.join(sorted(p.name for p in (Path(out) / 'onnx').iterdir()))}")
