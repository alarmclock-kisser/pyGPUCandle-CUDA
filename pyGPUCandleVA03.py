# IMPORTS
import os
import wave
import torchaudio
from torchaudio import functional as F
from torchaudio import transforms as T
import pyperclip

# VALUES


# FUNCTIONS
def load_from_clipboard(logging=False):
    if logging : print("----- load_from_clipboard(tensor, samplerate, factor, logging) -----")
    # Audiodaten aus der Zwischenablage laden
    data = pyperclip.paste()
    if logging : print('ZWISCHENABLAGE // CLIPBOARD: ', data[:10])
    # Samplerate erkennen
    samplerate = 48000  # Beispiel-Samplerate
    
    # Audio-Tensor und Samplerate zurückgeben
    return data, samplerate

def stretch_tensor(tensor, samplerate=48000, factor=1.000, logging=False):
    """Stretches the audio tensor using time stretching techniques."""
    if factor == 1.0:
        return tensor, samplerate

    if logging:
        print("----- stretch_tensor(tensor, samplerate, factor, logging) -----")

    # Create complex tensor
    n_fft = 1024
    hop_length = n_fft // 4
    window = torch.hann_window(n_fft, device=tensor.device)
    complex_tensor = torch.stft(tensor, n_fft=n_fft, hop_length=hop_length, return_complex=True, window=window)

    # Move to device (CUDA if available)
    device = torch.device(DEV)
    complex_tensor = complex_tensor.to(device)

    # Initialize stretch object
    stretch = T.TimeStretch(hop_length=hop_length, n_freq=n_fft // 2 + 1).to(device)

    # STFT -> Stretch -> ISTFT
    if logging:
        print(f"Stretching complex tensor by {factor}")
    stretched_complex = stretch(complex_tensor, factor)

    if logging:
        print(f"Adjusting with ISTFT by {n_fft}")
    original_length = tensor.shape[1]
    stretched_audio = torch.istft(stretched_complex, n_fft=n_fft, hop_length=hop_length, length=int(original_length / factor), window=window, center=True)

    if logging:
        print(f"Sample rate = {sample_rate}")

    return stretched_audio, samplerate

def main(factor=1.000, logging=True):
    print("----- main(factor, logging) -----")
    # Frühzeitiges Beenden, wenn die Zwischenablage leer ist
    if not pyperclip.paste():
        if logging : print('nichts in der Zwischenablage // empty clipboard')
        return
    else:
        print('Inhalt der Zwischenablage // Content of clipboard')
    
    # Audiodaten aus der Zwischenablage laden
    audio, samplerate = load_from_clipboard(logging)
    
    # Audio-Tensor strecken
    audio, samplerate = stretch_tensor(audio, samplerate, factor, logging)
    
    
    # Ergebnis in die Zwischenablage legen
    pyperclip.copy(audio)

if __name__ == "__main__":
    # Optionale Argumente entgegennehmen
    # (initbpm, goalbpm) XOR (factor) direkt XOR keine Argumente
    # Initial BPM und Ziel-BPM abfragen
    # Streckfaktor FAC berechnen

    main()
