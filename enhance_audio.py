import moviepy.editor as mp #Loads a video file as video clip object and useful for extracting audio
from pydub import AudioSegment #For processing audio file, helpful for normalizing and applying effects
import numpy as np
import scipy.signal as signal #for signal processing
from pydub.effects import normalize, compress_dynamic_range, low_pass_filter, high_pass_filter
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import librosa
import librosa.display
import soundfile as sf
import noisereduce as nr


# Loading the video
video = mp.VideoFileClip("/Users/joeppan/Desktop/Projects/AudioEnhancement/samples/sample.MP4")

# Extracting the  audio from the video
audio = video.audio
audio.write_audiofile("original_audio.wav")

# Load the extracted audio
audio_segment = AudioSegment.from_wav("original_audio.wav")


#Step-1 Noise Reduction- to reduce the low frequency noise
#The function removes or reduces low-frequency noise (like hums, rumbles, or static) from the input audio by applying a high-pass filter.
#Frequencies below the noise_level (500 Hz) are reduced, which helps eliminate unwanted background noise.
#The resulting audio data contains the same audio content as the original, but with cleaner and less noisy sound due to the removal of the low-frequency noise
#audio data is a numerical array extracted from audiosegment
#Noiselevel - any sound below is considered as noise 
#return: the filtered audio data as numpy array 
# Step-1 Noise Reduction - Reduce low-frequency noise
def multi_stage_filtering(audio_data):
    # High-pass filter to remove low-frequency noise
    b, a = signal.butter(3, 500 / (0.5 * 44100), btype='highpass')
    high_pass_filtered = signal.filtfilt(b, a, audio_data)

    # Low-pass filter to remove high-frequency hiss
    b, a = signal.butter(3, 8000 / (0.5 * 44100), btype='lowpass')
    low_pass_filtered = signal.filtfilt(b, a, high_pass_filtered)

    return low_pass_filtered

# Convert AudioSegment to numpy array and apply noise reduction
audio_data = np.array(audio_segment.get_array_of_samples())

# Apply noise reduction to the audio data
filtered_audio_data = multi_stage_filtering(audio_data)

# Normalize and ensure the data type is correct for AudioSegment conversion
filtered_audio_data = np.clip(filtered_audio_data, -32768, 32767)  # Clipping to int16 range
filtered_audio_data = filtered_audio_data.astype(np.int16)

# Convert the filtered audio data back to AudioSegment
filtered_audio_segment = AudioSegment(
    filtered_audio_data.tobytes(),
    frame_rate=audio_segment.frame_rate,
    sample_width=audio_segment.sample_width,
    channels=audio_segment.channels
)

# Export the filtered audio to a new WAV file
filtered_audio_segment.export("noisereduction_audio.wav", format="wav")

print("Filtered audio has been saved as 'noisereduction_audio.wav'.")

#comparing the old and new audio after doin the noise reduction part
# import matplotlib.pyplot as plt

# # Plotting the original audio waveform
# plt.figure(figsize=(12, 4))
# plt.plot(audio_data, color='b', alpha=0.6, label="Original Audio")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.title("Original Audio Waveform")
# plt.legend()
# plt.show()

# # Plotting filtered audio waveform
# plt.figure(figsize=(12, 4))
# plt.plot(filtered_audio_data, color='g', alpha=0.6, label="Filtered Audio")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.title("Filtered Audio Waveform")
# plt.legend()
# plt.show()

# # Plotting the  original vs filtered audio waveform for direct comparison
# plt.figure(figsize=(12, 4))
# plt.plot(audio_data, color='b', alpha=0.5, label="Original Audio")
# plt.plot(filtered_audio_data, color='g', alpha=0.7, label="Filtered Audio")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.title("Original vs Filtered Audio Waveform")
# plt.legend()
# plt.show()


# from scipy.signal import welch

# # Function to plot PSD
# def plot_psd(audio_data, sample_rate, title):
#     freqs, psd = welch(audio_data, sample_rate, nperseg=1024)
#     plt.figure(figsize=(10, 4))
#     plt.semilogy(freqs, psd)
#     plt.title(title)
#     plt.xlabel('Frequency [Hz]')
#     plt.ylabel('Power/Frequency (dB/Hz)')
#     plt.grid()
#     plt.show()

# # Plot PSD for original audio
# plot_psd(audio_data, audio_segment.frame_rate, "Original Audio Power Spectral Density")

# # Plot PSD for filtered audio
# plot_psd(filtered_audio_data, audio_segment.frame_rate, "Filtered Audio Power Spectral Density")


# from pydub.playback import play

# # Play the original audio
# print("Playing original audio...")
# play(audio_segment)

# # Play the filtered audio
# print("Playing filtered audio...")
# play(filtered_audio_segment)


# #Step-2 Applying the compression to the filtered audio
# compressed_audio_segment = compress_dynamic_range(
#     filtered_audio_segment,
#     threshold=-25.0,  
#     ratio=3.5,        
#     attack=10.0,      
#     release=100.0    
# )


# # Exporting  the compressed audio to a new WAV file
# compressed_audio_segment.export("compressed_noisereduction_audio.wav", format="wav")

# print("Compressed filtered audio has been saved as 'compressed_noisereduction_audio.wav'.")


#checking how much noise is reduced
# Function to calculate SNR
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:  # Prevent division by zero
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

# # Calculating the noise by subtracting filtered signal from the original signal
# noise = audio_data - filtered_audio_data

# # Calculate SNR before and after noise reduction
# snr_before = calculate_snr(audio_data, noise)  
# snr_after = calculate_snr(filtered_audio_data, noise) 

# # # Print SNR results
# # print(f"SNR Before Noise Reduction: {snr_before:.2f} dB")
# # print(f"SNR After Noise Reduction: {snr_after:.2f} dB")




#step-3 de-reverberation to reduce echo or reverb that may be present in old audio recordings, especially in noisy environment
# Define the Wiener filter function with custom window size for de-reverberation
# Step 1: Normalize audio data to floating-point range (-1.0 to 1.0)
def normalize_audio(original_audio, dereverb_audio):
    original_max = np.max(np.abs(original_audio))
    dereverb_max = np.max(np.abs(dereverb_audio))
    if dereverb_max == 0:
        return dereverb_audio
    scaling_factor = original_max / dereverb_max
    return dereverb_audio * scaling_factor

# Step 2: Signal-to-Reverberation Ratio (SRR) calculation function
def calculate_srr(original_signal, dereverb_signal):
    original_power = np.mean(np.square(original_signal))
    reverb_residual = original_signal - dereverb_signal
    reverb_power = np.mean(np.square(reverb_residual))
    return 10 * np.log10(original_power / reverb_power)

# Step 3: Apply de-reverberation using Wiener filter 
def wiener_de_reverb(audio_data, window_size=5):
    de_reverb_audio = signal.wiener(audio_data, mysize=window_size)
    return de_reverb_audio

# Step 4: De-reverberation using noisereduce with lower aggressiveness
def apply_noisereduce(audio_data_float, sr, prop_decrease_value=0.2):
    de_reverberated_audio = nr.reduce_noise(
        y=audio_data_float, 
        sr=sr, 
        stationary=False, 
        prop_decrease=prop_decrease_value
    )
    return de_reverberated_audio

# Load original audio segment
audio_segment = AudioSegment.from_wav("original_audio.wav")

# Convert the AudioSegment to numpy array and normalize to float32
filtered_audio_data = np.array(audio_segment.get_array_of_samples())
filtered_audio_data_float = filtered_audio_data.astype(np.float32) / 32768.0  # Normalizing to float32 range (-1.0 to 1.0)

# Step 5: Apply de-reverberation using noisereduce with lower aggressiveness
de_reverberated_audio = apply_noisereduce(filtered_audio_data_float, sr=audio_segment.frame_rate, prop_decrease_value=0.2)

# Step 6: Normalize the dereverberated audio to match original signal amplitude
de_reverberated_audio_normalized = normalize_audio(filtered_audio_data_float, de_reverberated_audio)

# Step 7: Convert back to int16 format for saving
de_reverberated_audio_int16 = (de_reverberated_audio_normalized * 32767).astype(np.int16)

# Step 8: Convert to AudioSegment and save the result
de_reverb_audio_segment = AudioSegment(
    de_reverberated_audio_int16.tobytes(),
    frame_rate=audio_segment.frame_rate,
    sample_width=audio_segment.sample_width,
    channels=audio_segment.channels
)

# Export the de-reverberated audio to a WAV file
de_reverb_audio_segment.export("noisereduce_de_reverb_audio.wav", format="wav")
print("De-reverberated audio (noisereduce) has been saved as 'noisereduce_de_reverb_audio.wav'.")

# Step 9: Calculate SRR before and after de-reverberation
srr_before = calculate_srr(filtered_audio_data_float, filtered_audio_data_float)  # Original SRR (compared to itself)
srr_after = calculate_srr(filtered_audio_data_float, de_reverberated_audio_normalized)  # SRR after de-reverberation

print(f"SRR Before De-reverberation: {srr_before:.2f} dB")
print(f"SRR After De-reverberation: {srr_after:.2f} dB")

#de-reverberation process worked effectively reducing the reverb while maintaining the integrity of the original signal
#SRR After De-reverberation is a positive value, which indicates that the processed audio is much cleaner compared to the original audio in terms of reverberation.
#SRR improvement to 24.76 dB shows that the de-reverberation process was effective. Cleaned up the audio, reducing reverb while maintaining the core signal.


# Audion restoration- to reduce specific types of degradation  such as pops clicks, crackle or vinyl record noises  in the audio
