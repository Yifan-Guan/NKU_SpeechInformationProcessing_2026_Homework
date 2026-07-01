import os
import librosa  
import matplotlib.pyplot as plt  
import IPython.display as ipd
import pandas as pd
import numpy as np

def draw_waveform(audio_file, save_dir):
    y, sr = librosa.load(audio_file)
    pd.Series(y).plot(figsize=(10, 5), lw=1, title=f'{audio_file}: Raw Audio Example')
    plt.savefig(f"{save_dir}/{audio_file.split('/')[-1].split('.')[0]}_waveform.png")
    plt.close()

def draw_spectrogram(audio_file, save_dir):
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 5))
    img = librosa.display.specshow(S_db,
                                   x_axis='time',
                               y_axis='log',
                                   ax=ax)
    ax.set_title('Spectogram Example', fontsize=20)
    fig.colorbar(img, ax=ax, format=f'%0.2f')
    plt.savefig(f"{save_dir}/{audio_file.split('/')[-1].split('.')[0]}_spectrogram.png")
    plt.close()

def draw_mel_spectrogram(audio_file, save_dir):
    y, sr = librosa.load(audio_file)
    n_fft = 2048       # FFT 大小
    hop_length = 512   # 帧移
    n_mels = 128       # Mel 频带数
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 5))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel")
    plt.colorbar(label="Decibels (dB)")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Frequency")
    plt.title("Mel Spectrogram (librosa)")
    plt.savefig(f"{save_dir}/{audio_file.split('/')[-1].split('.')[0]}_mel_spectrogram.png")
    plt.close()

if __name__ == "__main__":
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # audio_files = ["samples/speaker1.wav", 
    #                "samples/speaker2.wav",
    #                "samples/speaker1_rec.wav",
    #                "samples/speaker2_rec.wav"]

    audio_files = ["samples/speaker.wav",
                   "samples/speaker1_timbre_rec.wav",]

    for audio_file in audio_files:
        print(f"Processing {audio_file}...")
        draw_waveform(audio_file, save_dir)
        print(f"Waveform saved for {audio_file}.")
        draw_spectrogram(audio_file, save_dir)
        print(f"Spectrogram saved for {audio_file}.")
        draw_mel_spectrogram(audio_file, save_dir)
        print(f"Mel spectrogram saved for {audio_file}.")