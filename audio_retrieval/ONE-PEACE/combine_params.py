from collections import OrderedDict
from transformers import RobertaModel
# from one_peace.model import OnePeaceModel
import torch
from one_peace.models import OnePeaceRetrievalModel, OnePeaceBaseModel, OnePeacePretrainModel
from one_peace.models.one_peace.one_peace_retrieval import OnePeaceRetrievalConfig
from one_peace.models.one_peace.one_peace_pretrain import OnePeacePretrainConfig
from one_peace.data.pretrain_data.audio_text_pretrain_dataset import AudioTextPretrainDataset

import torch.nn as nn
from fairseq.data import Dictionary
import os

# RoBERTa 모델 로드
# roberta_model = RobertaModel.from_pretrained("roberta-base")
# onepeace_model = torch.load("/workspace/jaeyoung/checkpoints/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt")
roberta_model = RobertaModel.from_pretrained("roberta-large")
onepeace_model = torch.load("/workspace/jaeyoung/onepeace_pretrained_chkpoint/finetune_al_retrieval_onepiece.pt")
# onepeace_model = torch.load("/workspace/jaeyoung/onepeace_pretrained_chkpoint/one-peace.pt")


# Define mapping function
def map_roberta_to_onepeace(roberta_params):
    mapping = {
    "embeddings.word_embeddings.weight": "encoder_wrapper.text_adapter.embed_tokens.weight",
    "embeddings.position_embeddings.weight": "encoder_wrapper.text_adapter.embed_positions.weight",
    "embeddings.token_type_embeddings.weight": "encoder_wrapper.text_adapter.token_type_embeddings.weight",
    # "embeddings.LayerNorm.weight": "encoder_wrapper.text_adapter.LayerNorm.weight",
    # "embeddings.LayerNorm.bias": "encoder_wrapper.text_adapter.LayerNorm.bias"
    }

    # for i in range(23):
    for i in range(10, 24, 2):
        onepeace_index = round(i * 1.7)
        mapping.update({
            # Mapping self-attention components for each layer
            f"encoder.layer.{i}.attention.self.query.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.q_proj.weight",
            f"encoder.layer.{i}.attention.self.query.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.q_proj.bias",
            f"encoder.layer.{i}.attention.self.key.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.k_proj.weight",
            f"encoder.layer.{i}.attention.self.key.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.k_proj.bias",
            f"encoder.layer.{i}.attention.self.value.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.v_proj.weight",
            f"encoder.layer.{i}.attention.self.value.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.v_proj.bias",
            f"encoder.layer.{i}.attention.output.dense.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.out_proj.weight",
            f"encoder.layer.{i}.attention.output.dense.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn.out_proj.bias",
            f"encoder.layer.{i}.attention.output.LayerNorm.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn_layer_norm.weight",
            f"encoder.layer.{i}.attention.output.LayerNorm.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.self_attn_layer_norm.bias",
            # # Mapping feed-forward network components for each layer    
            f"encoder.layer.{i}.intermediate.dense.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.text_ffn.0.wi_0.weight",
            f"encoder.layer.{i}.intermediate.dense.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.text_ffn.0.wi_0.bias",
            f"encoder.layer.{i}.output.dense.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.text_ffn.2.weight",
            f"encoder.layer.{i}.output.dense.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.text_ffn.2.bias",
            # f"encoder.layer.{i}.output.LayerNorm.weight": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.final_layer_norm.weight",
            # f"encoder.layer.{i}.output.LayerNorm.bias": f"encoder_wrapper.fusion_model.layers.{onepeace_index}.final_layer_norm.bias"
        })

    converted_params = OrderedDict()
    for key, value in mapping.items():
        if key in roberta_params:
            converted_params[value] = roberta_params[key]
    return converted_params

# Ensure dimensions match
def match_dimensions(tensor, target_tensor):
    # Adjust for tensor dimensionality
    if tensor.dim() == 1:
        current_feature_size = 1
        tensor = tensor.unsqueeze(1)  # Convert to 2D by adding a dimension
    else:
        current_feature_size = tensor.shape[1]

    if target_tensor.dim() == 1:
        target_feature_size = 1
        target_tensor = target_tensor.unsqueeze(1)  # Convert to 2D by adding a dimension
    else:
        target_feature_size = target_tensor.shape[1]

    # Resize tensor if needed
    if tensor.shape[0] != target_tensor.shape[0]:
        # Handling vocabulary size mismatches
        difference = target_tensor.shape[0] - tensor.shape[0]
        if difference > 0:
            # Pad the smaller tensor
            padding = torch.zeros((difference, current_feature_size), device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)
        else:
            # Truncate the larger tensor
            tensor = tensor[:target_tensor.shape[0], :]

    # Transform feature dimension if necessary
    if current_feature_size != target_feature_size:
        linear_layer = nn.Linear(current_feature_size, target_feature_size).to(tensor.device)
        tensor = linear_layer(tensor)

    if tensor.dim() == 2 and (current_feature_size == 1 or target_feature_size == 1):
        tensor = tensor.squeeze(1)  # Remove the artificial dimension if original was 1D

    return tensor


# Merge parameters
def merge_parameters(onepeace_params, mapped_params, alpha=0.4):
    combined_params = OrderedDict()
    for key in onepeace_params:
        if key in mapped_params:
            matched_param = match_dimensions(mapped_params[key], onepeace_params[key])
            combined_params[key] = alpha * matched_param + (1 - alpha) * onepeace_params[key]
        else:
            combined_params[key] = onepeace_params[key]
    return combined_params

import torch

# Function to clone the initial state of model parameters
def clone_model_state(model):
    return {name: param.clone() for name, param in model.state_dict().items()}


# Function to compare and print changes in the model parameters
def print_updated_params(model, initial_state):
    for name, param in model.state_dict().items():
        initial_param = initial_state[name]
        # Using 'torch.equal' to check if two tensors are the same
        if not torch.equal(param, initial_param):
            print(f"Parameter updated: {name}")




if __name__=='__main__':
    roberta_params = roberta_model.state_dict()
    onepeace_params = onepeace_model['model']
    onepeace_config = onepeace_model['cfg']  # Safely load config if it exists

    mapped_params = map_roberta_to_onepeace(roberta_params)
    combined_params = merge_parameters(onepeace_params, mapped_params)

    # Verify the combined parameters
    print(combined_params.keys())
    dictionary = Dictionary.load(os.path.join('/workspace/jaeyoung/dcase2024_retrieval/ONE-PEACE/one_peace/utils/BPE', "dict.txt"))

    # onepeace_model = OnePeaceRetrievalModel(OnePeaceRetrievalConfig, src_dict=dictionary, head_type='al')

    # # Update ONE-PEACE model with combined parameters
    # onepeace_model.load_state_dict(combined_params, strict=False)
    model_path = '/workspace/jaeyoung/onepeace_pretrained_chkpoint/onepeace_roberta_l_ensemble40_layer13_23.pt'
    torch.save({'model':combined_params, 'cfg':onepeace_config}, model_path)

    # Print success message
    print("ONE-PEACE model's language branch successfully initialized with combined parameters.")
