import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
import twi_model as tm


def test_clean_text_keeps_twi_letters_and_tidies_quotes():
    assert tm.clean_text("  Ɛte  sɛn?\n") == "Ɛte sɛn?"
    assert tm.clean_text("Ɔbaa no ‘kɔɔ’ fie") == "Ɔbaa no 'kɔɔ' fie"
    # composed and decomposed forms come out the same
    assert tm.clean_text("é") == tm.clean_text("é")


def test_normalize_for_wer():
    assert tm.normalize_for_wer("Mepa wo kyɛw, ma me nsuo!") == "mepa wo kyɛw ma me nsuo"


def test_find_text_column():
    assert tm.find_text_column(["audio", "transcription", "speaker"]) == "transcription"
    assert tm.find_text_column(["audio", "sentence"]) == "sentence"
    assert tm.find_text_column(["audio", "path"]) is None


def test_find_pair_columns():
    _, a, b = tm.find_pair_columns({"Twi": "Ɛte sɛn?", "English": "How are you?"})
    assert (a, b) == ("Twi", "English")
    row, a, b = tm.find_pair_columns({"translation": {"tw": "Ɛte sɛn?", "en": "How are you?"}})
    assert (row[a], row[b]) == ("Ɛte sɛn?", "How are you?")
    assert tm.find_pair_columns({"text": "x", "label": 1})[1] is None


def test_language_slot_is_what_the_page_asks_for():
    page = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text(encoding="utf-8")
    assert f"opts.language = '{tm.LANGUAGE_TOKEN}'" in page
