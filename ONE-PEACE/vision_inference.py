import torch
import json
from one_peace.models import from_pretrained
from tqdm import tqdm
import pandas as pd
import os
from glob import glob

# TODO:: al rep model text parameters로 update한 vl model 로 추론 체크
device = "cuda" if torch.cuda.is_available() else "cpu"
model = from_pretrained(
    # model_name_or_path="/workspace/jaeyoung/checkpoints/one_peace/mmtts_vl_0907_lr2e-6/checkpoint_best.pt",
    model_name_or_path="/workspace/jaeyoung/checkpoints/one-peace_pretrained.pt",
    model_type="one_peace_retrieval",
    device=device,
    dtype="float16"
)


def find_jpg_files(root_dir):
    return glob(os.path.join(root_dir, "**/*.jpg"), recursive=True)


# Load captions and prepare audio files
captions_path = "/workspace/jaeyoung/StoryTeller/valid1000_merged_caption_MMTTS.csv"
image_dir = "/workspace/jaeyoung/datasets/mm-tts-dataset/video_image_save"
# image_dir = "/workspace/jaeyoung/StoryTeller/ONE-PEACE/assets"
df = pd.read_csv(captions_path)
# text_queries = df['caption1'].tolist()[:10]
text_queries = ['A female said with her sorrowful eyes.']
# text_queries = ['An animal barking.']
image_list = find_jpg_files(image_dir)[6000:9000]
# image_list = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.jpg') or x.endswith('JPEG')]

# Prepare results dataframe
results_df = pd.DataFrame(columns=['caption', 'fname_1', 'fname_2', 'fname_3', 'fname_4', 'fname_5', 'fname_6', 'fname_7', 'fname_8', 'fname_9', 'fname_10'])

# Batch processing parameters
audio_batch_size = 10  # Adjust based on your GPU capacity
text_batch_size = 50   # Adjust based on your GPU capacity

if __name__ == '__main__':
    with torch.no_grad():
        # Process audio in batches
        all_image_features = []
        for i in range(0, len(image_list), audio_batch_size):
            batch_image_list = image_list[i:i + audio_batch_size]
            src_images = model.process_image(batch_image_list)
            batch_image_features = model.extract_image_features(src_images)
            all_image_features.append(batch_image_features)
        all_image_features = torch.cat(all_image_features, dim=0)

        # Process text in batches
        all_text_features = []
        for i in range(0, len(text_queries), text_batch_size):
            batch_text_queries = text_queries[i:i + text_batch_size]
            text_tokens = model.process_text(batch_text_queries)
            batch_text_features = model.extract_text_features(text_tokens)
            all_text_features.append(batch_text_features)
        all_text_features = torch.cat(all_text_features, dim=0)

        # Compute similarity scores between all text features and all audio features
        similarity_scores = torch.matmul(all_image_features, all_text_features.T)

        # Retrieve top matching audio files for each text query
        for text_idx, single_text_features in tqdm(enumerate(all_text_features)):
            single_similarity_scores = similarity_scores[:, text_idx]
            top_audio_indices = torch.topk(single_similarity_scores, k=5).indices
            top_audio_indices = top_audio_indices.cpu().numpy().tolist()
            top_files = [image_list[idx] for idx in top_audio_indices]
            # results_df.loc[len(results_df)] = [text_queries[text_idx]] + top_files
            print(f'text_queries:{text_queries} || top_files:{top_files}')
            results_df.loc[len(results_df)] = [text_queries[text_idx]] + [file for file in top_files]
            
        # Save results to CSV
        # results_csv_path = '/workspace/jaeyoung/StoryTeller/inferred_image_caption_MMTTS.csv'
        # results_df.to_csv(results_csv_path, index=False)
        # print(f"Results saved to {results_csv_path}")
