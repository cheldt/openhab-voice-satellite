import numpy as np

from stt_proxy.tts import split_sentences, to_int16


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


def test_to_int16_scales_and_clips():
    samples = np.array([0.0, 0.5, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    pcm = to_int16(samples)
    assert pcm.dtype == np.int16
    assert pcm[0] == 0
    assert pcm[1] == 16383
    assert pcm[2] == 32767
    assert pcm[3] == -32767
    assert pcm[4] == 32767  # clipped
    assert pcm[5] == -32767  # clipped
