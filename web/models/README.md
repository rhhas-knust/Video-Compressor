# On-device models

`notes-model/` holds the trained "important points" model used by the
Transcribe & notes tool. It is optional: without it the page picks notes with
its built-in rules.

Train it with `training/train_notes_model.ipynb` in Google Colab, then upload the
folder it produces here, keeping this layout:

```
web/models/notes-model/
  config.json
  tokenizer.json
  tokenizer_config.json
  onnx/model_quantized.onnx
```
