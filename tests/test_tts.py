from stt_proxy.tts import split_sentences


def test_split_basic():
    assert split_sentences("Hallo. Wie geht es dir? Gut!") == [
        "Hallo.", "Wie geht es dir?", "Gut!",
    ]


def test_split_single_sentence():
    assert split_sentences("Das Licht ist an") == ["Das Licht ist an"]


def test_split_empty():
    assert split_sentences("   ") == []


def test_split_keeps_abbrev_number_text_together():
    # no split without whitespace after the punctuation
    assert split_sentences("Es ist 21.30 Uhr.") == ["Es ist 21.30 Uhr."]
