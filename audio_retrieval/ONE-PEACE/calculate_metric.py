import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from tqdm import tqdm
from one_peace.models import from_pretrained

# Initialize device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/workspace/jaeyoung/checkpoints/onepeace_roberta_l_ensemble40_layer13_23_0705/checkpoint4.pt"
print(f'model: {model_path}')
model = from_pretrained(
    model_name_or_path=model_path,
    # model_name_or_path="/workspace/jaeyoung/onepeace_pretrained_chkpoint/retrieval_onepeace_roberta_l_ensemble40.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta40_pretrained_fusion_rep_0701/checkpoint_best.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta_finetuned_0701_v2/checkpoint_best.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta40_fusion_rep_unfz_txt_fusion_0703/checkpoint_best.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta70_fusion_rep_unfz_txt_fusion_0703_lr-4/checkpoint_best.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta70_fusion_rep_unfz_txt_fusion_0703_lr_4_lr_6/checkpoint_best.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta_finetuned_fusion_rep_0701/checkpoint6.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_roberta40_pretrained_fusion_rep_0701/checkpoint_best.pt",
    # model_name_or_path="/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt",
    # model_name_or_path = "/workspace/jaeyoung/checkpoints/onepeace_finetuned_middle_clotho/checkpoint2.pt",
    # model_name_or_path= "/workspace/jaeyoung/checkpoints/onepeace_finetuned_middle_clotho_metric_audio_r1/checkpoint_best.pt",
    model_type="one_peace_retrieval",
    device=device,
    dtype="float16"
)

# Load captions and prepare audio files
captions_path = "/workspace/jaeyoung/data/clotho_dataset/clotho_captions_evaluation.csv"
# captions_path = "/workspace/jaeyoung/data/clotho_dataset/merged_clotho_val_nf1.csv"
audio_dir = "/workspace/jaeyoung/data/clotho_dataset/evaluation_16k"
df = pd.read_csv(captions_path)
text_queries = df['caption_1'].tolist()
file_names = df['file_name'].tolist()
audio_files = os.listdir(audio_dir)
audio_list = [os.path.join(audio_dir, x) for x in audio_files if not x.startswith('._')]

# Map file names to indices in the audio_list for faster retrieval
audio_index_map = {os.path.basename(file): i for i, file in enumerate(audio_list)}

# Debugging: Check if file names from df match those in audio_index_map
print("Checking file name alignment:")
for file_name in file_names:
    if os.path.basename(file_name) not in audio_index_map:
        print(f"Missing file: {file_name}")

# Batch processing parameters
audio_batch_size = 10
text_batch_size = 50

if __name__ == '__main__':
    with torch.no_grad():
        all_audio_features = []
        all_text_features = []

        # Process audio in batches
        for i in range(0, len(audio_list), audio_batch_size):
            batch_audio_list = audio_list[i:i + audio_batch_size]
            src_audios, audio_padding_masks = model.process_audio(batch_audio_list)
            batch_audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
            all_audio_features.append(batch_audio_features)
        
        all_audio_features = torch.cat(all_audio_features, dim=0).to(device)

        # Process text in batches
        for i in range(0, len(text_queries), text_batch_size):
            batch_text_queries = text_queries[i:i + text_batch_size]
            text_tokens = model.process_text(batch_text_queries)
            batch_text_features = model.extract_text_features(text_tokens)
            all_text_features.append(batch_text_features)
        
        all_text_features = torch.cat(all_text_features, dim=0).to(device)

        # Compute similarity scores
        similarity_scores = torch.matmul(all_text_features, all_audio_features.T)

        target_indices = torch.tensor([audio_index_map.get(os.path.basename(file_name), -1) for file_name in file_names], device=device)

        # Compute recall and mAP
        top_one = similarity_scores.topk(1, dim=1).indices
        top_five = similarity_scores.topk(5, dim=1).indices
        top_ten = similarity_scores.topk(10, dim=1).indices

        r_1 = (top_one.squeeze(1) == target_indices).float().mean().item()
        r_5 = (top_five == target_indices.unsqueeze(1)).any(dim=1).float().mean().item()
        r_10 = (top_ten == target_indices.unsqueeze(1)).any(dim=1).float().mean().item()

        print(f"R@1: {r_1}, R@5: {r_5}, R@10: {r_10}")

        # Additional debug to understand AP calculation
        ap_scores = []
        for i, target in enumerate(target_indices):
            if target == -1:
                continue  # Skip missing files
            valid_mask = (top_ten[i] == target).nonzero(as_tuple=True)
            if valid_mask[0].numel() > 0:
                rank = valid_mask[0][0].item() + 1
                ap_scores.append(1 / rank)
            else:
                ap_scores.append(0)
        mAP = sum(ap_scores) / len(ap_scores) if ap_scores else 0

        print(f"Mean AP: {mAP}")