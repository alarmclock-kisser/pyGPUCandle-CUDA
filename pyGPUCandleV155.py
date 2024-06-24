# Import necessary libraries
import os
import argparse
import torch
import torchaudio
from torchaudio import functional as F
from torchaudio import transforms as T
import sounddevice as SD
import numpy as NP
import soundfile as SF
import threading
import librosa

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Initialize default (global) variables
LOG = True
FIL = "Z:\\Your\\Path\\to\\File.mp3"  # Default audio file path, Replace Z with your drive and the rest
PTH = os.path.join(os.path.dirname(FIL), "stretched")  # Default output directory for stretched audio, creates subfulder to Standard-File
FOR = (".mp3", ".wav", ".flac")  # Supported audio file formats, add more if using differend backend
SAM = 48000  # Target sample rate, standard for mp3
FAC = 1.000  # Stretch factor, standard for not stretching anything
DEV = "cuda" if torch.cuda.is_available() else "cpu"  # Determine device (CUDA or CPU)
BPM = 100.0  # Default BPM

# Functions
def verify_file(filepath=FIL, supported_formats=FOR, logging=LOG):
    """Verifies the audio file path and returns it if valid, or the default path if not."""
    if logging:
        print("----- verify_file(filepath, logging) -----")

    if not filepath or not os.path.exists(filepath):
        if logging:
            print("No valid filepath was given. Using standard-filepath:", FIL)
        return FIL

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in supported_formats:
        if logging:
            print(f"Extension {ext} is not supported, see: {FOR}.")
            print(f"Using standard-filepath: {FIL}")
        return FIL

    if logging:
        print(filepath, "is a valid filepath.")
    return filepath

def verify_directory(directory=PTH, logging=LOG):
    """Verifies the output directory and creates it if it doesn't exist."""
    if logging:
        print("----- verify_directory(directory, logging) -----")

    directory = os.path.abspath(directory)
    if logging:
        print(f"Absolute directory path: {directory}")

    if not os.path.exists(directory):
        os.makedirs(directory)
        if logging:
            print(f"Directory created: {directory}")
    else:
        if logging:
            print("Directory already exists: ", directory)
    
    return directory

def load_audio(filepath=FIL, logging=LOG):
    """Loads audio data from the given path, calculates BPM, and preprocesses it."""
    if logging:
        print("----- load_audio(filepath, logging) -----")

    waveform, sample_rate = torchaudio.load(filepath)
    waveform = waveform.to(DEV)
    waveform = waveform / torch.max(torch.abs(waveform))  # Normalize

    # Handle channels
    num_channels = waveform.shape[0]
    if num_channels == 1:
        if logging:
            print("\nMono-audio detected. Doubling channel to stereo.")
        waveform = torch.cat((waveform, waveform), dim=0)
    elif num_channels > 2:
        if logging:
            print(f"\n{num_channels}-channel-audio detected. Converting to stereo (2).")
        waveform = waveform[:2, :]  # Use first two channels

    if waveform.shape[0] != 2:
        raise ValueError("\nInvalid number of channels detected after normalization. Expected stereo-format (2 channels).")

    bpm = float(scan_bpm(filepath, logging))  # Calculate BPM

    if logging:
        print("Audio-data loaded and processed:")
        print(f"  Shape: {waveform.shape}")
        print(f"  Sample Rate: {sample_rate} Hz")
        print(f"  Device: {DEV}")

    return waveform, sample_rate, bpm

def scan_bpm(filepath=FIL, logging=LOG):
    """Scans BPM of a track."""
    if logging:
        print("----- scan_bpm(filepath, logging) -----")

    y, sr = librosa.load(filepath)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    tempo = float(tempo[0])
    if logging:
        print(f"Calculated BPM: {tempo:.4f}")

    return tempo

def stretch_tensor(tensor, sample_rate=SAM, factor=FAC, logging=LOG):
    """Stretches the audio tensor using time stretching techniques."""
    if factor == 1.0:
        return tensor, sample_rate

    if logging:
        print("----- stretch_tensor(tensor, sample_rate, factor, logging) -----")

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

    return stretched_audio, sample_rate

def play_audio(tensor, sample_rate=SAM, logging=LOG):
    """Plays tensor on sounddevice."""
    if logging:
        print("----- play_audio(tensor, sample_rate, logging) -----")

    if not isinstance(tensor, torch.Tensor):
        if logging:
            print("Error: Input is no torch-tensor, cannot process data-type.")
        return

    audio_np = tensor.cpu().numpy()
    num_channels = audio_np.shape[0]

    if logging:
        print(f"{num_channels}-channel-audio detected.")

    if num_channels > 2:
        if logging:
            print(f"Warning: Audio has {num_channels} channels. Only playing first both.")
        audio_np = audio_np[:2, :]

    try:
        SD.play(audio_np.T, sample_rate)
        if logging:
            print(f"Audio is playing on: {DEV}")
        input("Press ENTER to stop...")
        SD.stop()
    except Exception as e:
        if logging:
            print(f"Error playing audio: {e}")

def save_audio(tensor, sample_rate=SAM, directory=PTH, logging=LOG):
    """Saves stretched tensor with factor / bpm to file."""
    if logging:
        print("\n\n----- save_audio(tensor, sample_rate, directory, logging) -----")

    if tensor is None or sample_rate is None:
        if logging:
            print("Error tensor or samplerate invalid.")
        return

    if FAC == 1.000:  # Don't save if no stretching was applied
        if logging:
            print("Stretching factor equals 1.000, not saving anything, same data.")
        return FIL  # Return original filepath

    # Extract base filename and extension
    base_filename, ext = os.path.splitext(os.path.basename(FIL))

    # Insert stretch factor into filename
    if FAC < 5 :
        new_filename = f"{base_filename}_{FAC:.3f}x{ext}"
    if FAC > 10 :
        new_filename = f"{base_filename}_{FAC:.2f}BPM{ext}"

    filepath = os.path.join(directory, new_filename)

    if logging:
        print(f"Saving data to path: {filepath}")

    directory = os.path.dirname(filepath)
    verify_directory(directory, logging)  # Ensure directory exists

    SF.write(filepath, tensor.cpu().numpy().T, sample_rate)
    return filepath

def main(filepath=FIL, directory=PTH, stretch=FAC, logging=LOG):
    """main-function."""
    if logging:
        print("\n\n----- main-function main(filepath, directory, factor, logging) -----")

    # Verify paths
    filepath = verify_file(filepath, logging=logging)
    directory = verify_directory(directory, logging=logging)

    # loading audio and bpm
    waveform, sample_rate, bpm = load_audio(filepath, logging=logging)

    # adjusting factor or bpm stretch rate
    if stretch >= 10.0:
        if bpm is not None:
            stretch = stretch / bpm
            if logging:
                print(f"New stretching factor after detecting BPM: {stretch:.3f}")
        else:
            if logging:
                print("Error detecting BPM. Stretching set to standard 1.000.")
            stretch = 1.000

    # actual stretching
    stretched_tensor, stretched_sample_rate = stretch_tensor(waveform, sample_rate, stretch, logging=logging)

    # saving new audio if factor != 1.000
    if stretch != 1.0:
        filepath = save_audio(stretched_tensor, stretched_sample_rate, directory, logging=logging)

    # playing result
    play_audio(stretched_tensor, stretched_sample_rate, logging=logging)
    
    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Stretching Tool")
    parser.add_argument("-f", "--file", type=str, help="path to audio-file")
    parser.add_argument("-o", "--output_dir", type=str, help="output directory")
    parser.add_argument("-s", "--stretch", type=float, help="stretching factor or desired BPM (0.1 to 2.0 or greater 10 for bpm stretching)")
    parser.add_argument("-l", "--logging", action="store_true", help="logging?")
    args = parser.parse_args()

    # If no command line arguments are provided, ask for user input
    if not any(vars(args).values()):
        FIL = verify_file(input('filepath?: ').strip('"'))

        PTH = verify_directory(input("\noutput directory?: ")) if input("\noutput directory?: ") else verify_directory(os.path.join(os.path.dirname(FIL), "stretched"))

        while True:
            try:
                FAC = float(input("\nstretching factor OR desired new BPM?: "))
                if 0.099 <= FAC:
                    break
                else:
                    print("factor has to be 0.100 OR greater.")
            except ValueError:
                print("Invalid input, enter FLOAT.")

        LOG = input('\nactivate logging? (accept with y/j/1/yes): ').lower() in ["y", "yes", "1", "true", "ja", "j"]
        print("\n")

    # If command line arguments are provided, use them
    else:
        FIL = args.file
        PTH = args.output_dir
        FAC = args.stretch
        LOG = args.logging

    # Call the main function with the determined parameters
    output_filepath = main(filepath=FIL, directory=PTH, stretch=FAC, logging=LOG)

    if LOG:
        if output_filepath:
            print(f"Saved processed file at: {output_filepath}")
        print("----- end of script -----")
