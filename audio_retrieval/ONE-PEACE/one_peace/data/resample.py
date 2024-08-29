# import os
# import ray
# import librosa
# import numpy as np
# import soundfile as sf
# from tqdm import tqdm

# def list_audio_files(directory, ext='.flac'):
#     audio_files = []
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(ext) and not file.startswith('._'):
#                 audio_files.append(os.path.join(root, file))
#     return audio_files

# available_cpus = os.cpu_count()
# ray.init(num_cpus=available_cpus)

# @ray.remote
# def process_audio(input_file_path, output_directory, target_sr=16000):
#     try:
#         # Process the audio in chunks to manage memory usage
#         with sf.SoundFile(input_file_path) as sound_file:
#             sr = sound_file.samplerate
#             chunk_size = 10 * sr  # 10 seconds per chunk
#             audio_mono = []
#             while True:
#                 # Read a chunk of audio
#                 audio_chunk = sound_file.read(frames=chunk_size, dtype='float32', always_2d=False)
#                 if audio_chunk.shape[0] == 0:
#                     break  # End of file
#                 if audio_chunk.ndim > 1:
#                     audio_chunk = np.mean(audio_chunk, axis=1)  # Convert to mono if stereo
#                 audio_mono.append(audio_chunk)
            
#             audio_mono = np.concatenate(audio_mono)

#             # Resample if necessary
#             if sr != target_sr:
#                 audio_resampled = librosa.resample(audio_mono, orig_sr=sr, target_sr=target_sr)
#             else:
#                 audio_resampled = audio_mono
            
#             # Construct the output file path
#             output_file_path = os.path.join(output_directory, os.path.basename(input_file_path))
#             # Save the resampled audio
#             sf.write(output_file_path, audio_resampled, target_sr)

#             print(f"Processed and saved {input_file_path} to {output_file_path}")
#     except Exception as e:
#         print(f"Error processing {input_file_path}: {str(e)}")

# if __name__ == '__main__':
#     input_directory = '/Volumes/One_Touch/audiocaps_set/filtered_audioset'
#     output_directory = '/Volumes/One_Touch/audiocaps_set/audioset_16k'
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)

#     audio_files = list_audio_files(input_directory)
#     futures = [process_audio.remote(input_path, output_directory) for input_path in tqdm(audio_files)]
#     ray.get(futures)

#     ray.shutdown()


import os
import ray
import librosa
import soundfile as sf
from tqdm import tqdm

# def list_audio_files(directory, ext='.flac'):
#     return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(ext)]


# def convert_stereo_to_mono(input_file_path, output_file_path):
#     try:
#         # Load the audio file as stereo
#         audio, sr = librosa.load(input_file_path, sr=None, mono=False)
        
#         # Convert to mono by averaging the two channels
#         if audio.ndim > 1:  # Check if audio is not already mono
#             audio_mono = librosa.to_mono(audio)
#         else:
#             audio_mono = audio  # It's already mono
        
#         # Save the mono audio file
#         sf.write(output_file_path, audio_mono, sr)
#         print(f"Converted {input_file_path} to mono and saved to {output_file_path}")
#     except Exception as e:
#         print(f"Error processing {input_file_path}: {str(e)}")


def list_audio_files(directory, ext='.wav'):
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(ext) and not file.startswith('._'):
                audio_files.append(os.path.join(root, file))
    return audio_files

available_cpus = os.cpu_count()
ray.init(num_cpus=available_cpus)  # Ray 초기화

@ray.remote
def resample_audio(input_file_path, output_directory, target_sr=16000):
    # 오디오 파일 읽기
    audio, sr = librosa.load(input_file_path, sr=None, mono=False)       
    # Convert to mono by averaging the two channels
    if audio.ndim > 1:  # Check if audio is not already mono
        audio_mono = librosa.to_mono(audio)
    else:
        audio_mono = audio  # It's already mono
    
    # 오디오 리샘플링
    if sr != target_sr:
        audio_resampled = librosa.resample(audio_mono, orig_sr=sr, target_sr=target_sr)
    else:
        audio_resampled = audio_mono
    
    # 리샘플된 오디오 파일 저장
    output_file_path = os.path.join(output_directory, os.path.basename(input_file_path))
    sf.write(output_file_path, audio_resampled, target_sr)


if __name__=='__main__':
    # 오디오 파일이 위치한 입력 폴더
    input_directory = '/Volumes/One_Touch/clotho/clotho_dataset/validation'
    # 리샘플된 오디오 파일을 저장할 출력 폴더
    output_directory = '/Volumes/One_Touch/clotho/clotho_dataset/validation_16k'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 폴더에서 오디오 파일 리스트 생성
    audio_files = list_audio_files(input_directory)

    # Ray를 사용하여 각 파일에 대해 리샘플링 작업을 비동기적으로 실행
    futures = [resample_audio.remote(input_path, output_directory) for input_path in tqdm(audio_files) if not os.path.basename(input_path).startswith('._')]
    ray.get(futures)  # 모든 작업이 완료될 때까지 대기

    ray.shutdown()  # Ray 종료
