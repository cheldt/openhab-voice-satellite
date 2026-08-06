import numpy as np

from openhab_voice_satellite.audio.chunker import FrameChunker


def _samples(n: int, start: int = 0) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.int16)


def test_exact_frame_yields_one_frame():
    chunker = FrameChunker(frame_samples=4)
    frames = chunker.push(_samples(4))
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], _samples(4))


def test_sub_frame_input_accumulates_across_pushes():
    chunker = FrameChunker(frame_samples=4)
    assert chunker.push(_samples(3)) == []
    frames = chunker.push(_samples(3, start=3))
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], _samples(4))


def test_multiple_frames_with_carry():
    chunker = FrameChunker(frame_samples=4)
    frames = chunker.push(_samples(10))
    assert [len(f) for f in frames] == [4, 4]
    np.testing.assert_array_equal(frames[0], _samples(4))
    np.testing.assert_array_equal(frames[1], _samples(4, start=4))
    # remainder (samples 8, 9) completes with the next push
    frames = chunker.push(_samples(2, start=10))
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], _samples(4, start=8))


def test_frames_are_owned_copies():
    # gst buffers are unmapped after the callback; frames must not alias input
    chunker = FrameChunker(frame_samples=2)
    data = _samples(2)
    frames = chunker.push(data)
    data[:] = -1
    np.testing.assert_array_equal(frames[0], _samples(2))


def test_dtype_preserved():
    chunker = FrameChunker(frame_samples=2)
    frames = chunker.push(_samples(2))
    assert frames[0].dtype == np.int16
