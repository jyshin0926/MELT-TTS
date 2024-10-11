import ray
import librosa
import soundfile as sf
import os

# Initialize Ray with a specific number of CPUs (example: 4 CPUs)
TOTAL_CPUS = 4
ray.init(num_cpus=TOTAL_CPUS)

@ray.remote(num_cpus=1)  # Each resampling task uses 1 CPU
def resample_audio(input_path, output_dir, target_sr=22050):
    try:
        # Load the audio file with the original sampling rate
        audio, sr = librosa.load(input_path, sr=None)

        # Check if resampling is needed
        if sr != target_sr:
            # Resample the audio
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        else:
            audio_resampled = audio  # No resampling needed

        # Prepare output path
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_resampled{ext}")

        # Save the resampled audio
        sf.write(output_path, audio_resampled, target_sr)

        return output_path
    except Exception as e:
        return f"Error processing {input_path}: {e}"

def resample_multiple_audios(input_dir, output_dir, target_sr=22050):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all audio files in the input directory with correct endswith usage
    audio_files = [
        os.path.join(input_dir, x)
        for x in os.listdir(input_dir)
        if x.lower().endswith(('.wav', '.flac'))
    ]

    if not audio_files:
        print("No audio files found in the input directory.")
        return []

    print(f"Found {len(audio_files)} audio files. Starting resampling...")

    # Launch resampling tasks in parallel
    resample_tasks = [
        resample_audio.remote(file, output_dir, target_sr) for file in audio_files
    ]

    # Collect the results as they complete
    resampled_files = ray.get(resample_tasks)

    print("Resampling completed. Resampled files:")
    for file in resampled_files:
        print(file)

    return resampled_files

if __name__ == "__main__":
	# Define input and output directories
	INPUT_DIR = '/workspace/jaeyoung/datasets/mm-tts-dataset/raw'
	OUTPUT_DIR = '/workspace/jaeyoung/datasets/mm-tts-dataset_resampled'
	TARGET_SAMPLING_RATE = 22050         # Target sampling rate in Hz

	# Start resampling
	resample_multiple_audios(INPUT_DIR, OUTPUT_DIR, TARGET_SAMPLING_RATE)

	# Shutdown Ray
	ray.shutdown()



# import ray
# import librosa
# import soundfile as sf
# import os

# ray.init(num_cpus=16)

# def resample_multiple_audio(input_dir, output_dir, tgt_sr=22050):
#     os.makedirs(output_dir, exist_ok=True)
#     audio_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.lower().endswith(('.wav', '.flac'))]
    
#     if not audio_files:
#         print('No audio files found in this dir')
#         return []
    
#     print(f"Found {len(audio_files)} audio files. Start resampling..")
    
#     resample_tasks = [resample_audio.remote(file, output_dir, tgt_sr) for file in audio_files]
#     resampled_files = ray.get(resample_tasks)
#     print('Resampling completed.')
    

# @ray.remote
# def resample_audio(input_path, output_dir, tgt_sr = 22050):
# 	try:
# 		audio, sr = librosa.load(input_path, sr=None)
# 		if sr != tgt_sr:
# 			audio_resampled = librosa.resample(audio, sr, tgt_sr)
# 		else:
# 			audio_resampled = audio
   
# 		filename = os.path.basename(input_path)
# 		name, ext = os.path.splitext(filename)
# 		output_path = os.path.join(output_dir, f"{name}_resampled{ext}")

# 		sf.write(output_path, audio_resampled, tgt_sr)

# 		return output_path
 
# 	except Exception as e:
# 		return f"Error {input_path}: {e}"

# if __name__=='__main__':
#     input_dir = '/workspace/jaeyoung/datasets/mm-tts-dataset/raw'
#     out_dir = '/workspace/jaeyoung/datasets/mm-tts-dataset_resampled'
#     resample_multiple_audio(input_dir, out_dir, 22050)
#     ray.shutdown()