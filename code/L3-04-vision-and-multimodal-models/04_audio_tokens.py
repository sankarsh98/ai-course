"""From waveform to tokens: a log-mel spectrogram with Whisper's front-end settings.

16 kHz audio, 25 ms windows (400 samples) every 10 ms (160 samples), 80 mel bands.
Whisper's encoder then applies two convolutions, the second with stride 2, so 30 s of
audio becomes 3000 frames and then 1500 encoder positions. NumPy only.
"""
import numpy as np

SR, WIN, HOP, N_MELS, N_FFT = 16000, 400, 160, 80, 512


def mel_filterbank(n_mels, n_fft, sr):
    hz_to_mel = lambda f: 2595 * np.log10(1 + f / 700)          # one common mel formula
    mel_to_hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = mel_to_hz(np.linspace(0, hz_to_mel(sr / 2), n_mels + 2))
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):                                     # triangular filters
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        fb[m] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    return fb, edges


def log_mel(wave):
    n_frames = 1 + (len(wave) - WIN) // HOP
    frames = np.stack([wave[i * HOP:i * HOP + WIN] for i in range(n_frames)]) * np.hanning(WIN)
    power = np.abs(np.fft.rfft(frames, n=N_FFT)) ** 2           # (frames, 257)
    fb, edges = mel_filterbank(N_MELS, N_FFT, SR)
    return np.log10(power @ fb.T + 1e-10), edges               # (frames, 80)


rng = np.random.default_rng(0)
t = np.arange(int(2.0 * SR)) / SR                               # 2 seconds
wave = np.where(t < 1.0, np.sin(2 * np.pi * 440 * t),           # 1 s of a 440 Hz tone, then
                np.sin(2 * np.pi * (300 * t + 1000 * (t - 1) ** 2)))  # a rising chirp
wave = wave + 0.01 * rng.standard_normal(len(t))
spec, edges = log_mel(wave)
centres = edges[1:-1]
print(f"2.0 s of audio = {len(wave)} samples -> log-mel {spec.shape} (frames x mel bands)")
print(f"mel band centres: first {centres[:3].round(0)} Hz ... last {centres[-2:].round(0)} Hz")
for sec in (0.5, 1.2, 1.5, 1.9):
    f = int(sec * SR / HOP)
    print(f"  t={sec:.1f}s  loudest band centre {centres[spec[f].argmax()]:6.0f} Hz")

print("\nsequence lengths a model has to attend over (arithmetic, not measured):")
whisper_frames = 30 * SR // HOP
print(f"  30 s of audio: {whisper_frames} frames -> {whisper_frames // 2} positions after the stride-2 conv")
print(f"  a 336x336 image in 14 px patches: {(336 // 14) ** 2} tokens; "
      f"after merging 2x2 patches: {(336 // 28) ** 2}")
fps, minutes = 2, 1
print(f"  {minutes} min of video at {fps} frames/s, 576 tokens per frame: {minutes * 60 * fps * 576:,} tokens")
