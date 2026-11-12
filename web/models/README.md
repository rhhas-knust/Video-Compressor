# On-device models

Models trained for this app, loaded from the site itself:

- `notes-model/`: the "important points" model used by Transcribe & notes.
  Optional: without it the page picks notes with its built-in rules.
- `twi-asr/` and `twi-en/`: Twi (Akan) speech recognition and Twi→English
  translation. Trained with `training/train_twi_model.ipynb`, which pushes
  them to a `twi-models` branch (their files are too big for GitHub's
  browser upload). Without them, Twi shows as not installed.

Train the notes model with `training/train_notes_model.ipynb` in Google Colab, then upload the
folder it produces here, keeping this layout:

```
web/models/notes-model/
  config.json
  tokenizer.json
  tokenizer_config.json
  onnx/model_quantized.onnx
```
