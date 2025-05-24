import torchaudio
import torch

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # mono
    return waveform, sample_rate

def get_mel_spec(file_path):
    waveform, _ = load_audio(file_path)
    mel = mel_spectrogram(waveform)
    return mel.squeeze().transpose(0, 1)  # (time, freq)
