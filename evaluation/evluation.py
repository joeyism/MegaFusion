import os
import json
import torch
import nicefid
import numpy as np
from PIL import Image
from collections import Counter
from pytorch_fid import fid_score
from nltk.translate import meteor_score
from transformers import CLIPProcessor, CLIPModel
from nltk.translate.bleu_score import sentence_bleu

import nltk
nltk.download('wordnet')

# Helper functions
def tokenize(sentence):
    return sentence.split()

def calculate_clip_similarity(image_path, text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        similarity = logits_per_image.softmax(dim=1).item()

    return similarity

def calculate_cider(reference, candidate):
    ref_tokens = tokenize(reference)
    can_tokens = tokenize(candidate)
    max_n = 4
    ref_ngrams = [tuple(ref_tokens[i:i+n]) for n in range(1, max_n+1) for i in range(len(ref_tokens)-n+1)]
    can_ngrams = [tuple(can_tokens[i:i+n]) for n in range(1, max_n+1) for i in range(len(can_tokens)-n+1)]
    ref_counts = Counter(ref_ngrams)
    can_counts = Counter(can_ngrams)
    cider_scores = []
    for n in range(1, max_n+1):
        common_ngrams = set(ref for ref in ref_counts if ref in can_counts)
        if len(common_ngrams) == 0:
            cider_scores.append(0.0)
        else:
            score = sum(min(ref_counts[ref], can_counts[ref]) for ref in common_ngrams)
            norm_score = score / len(can_ngrams)
            cider_scores.append(norm_score)
    geometric_mean = np.exp(np.mean(np.log(cider_scores)))
    return geometric_mean

def calculate_meteor(reference_caption, candidate_caption):
    reference_tokens = [reference_caption.split()]
    candidate_tokens = candidate_caption.split()
    meteor = meteor_score.meteor_score(reference_tokens, candidate_tokens)
    return meteor

def calculate_rouge(reference_caption, candidate_caption):
    rouge_score = sentence_bleu([reference_caption], candidate_caption)
    return rouge_score

# Evaluation functions
def evaluate_clip_similarity(config_file_path, generated_images_folder):
    clip_scores = 0
    i = 0
    with open(config_file_path, 'r') as file:
        coco_config = json.load(file)
    captions = coco_config['captions']
    for image_id, caption in captions.items():
        reference_caption = caption[0]
        image_file_name = f"{image_id}.png"  # Assuming images are in PNG format
        image_file_path = os.path.join(generated_images_folder, image_file_name)
        if os.path.exists(image_file_path):
            clip_score = calculate_clip_similarity(image_file_path, reference_caption)
            clip_scores += clip_score
            i += 1
    clip_scores /= i
    print(f"CLIP Image-Text Similarity Score: {clip_scores}")

def evaluate_cider(config_file_path, folder_path):
    cider_scores = 0
    i = 0
    with open(config_file_path, 'r') as file:
        coco_config = json.load(file)
    captions = coco_config['captions']
    for image_id, caption in captions.items():
        reference_caption = caption[0]
        txt_file_name = f"MS_{image_id}.txt"
        txt_file_path = os.path.join(folder_path, txt_file_name)
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                candidate_caption = file.read()
            cider_score = calculate_cider(reference_caption, candidate_caption)
            cider_scores += cider_score
            i += 1
    cider_scores /= i
    print("CIDEr Score:", cider_scores)

def evaluate_meteor(config_file_path, folder_path):
    meteor_score_values = 0
    i = 0
    with open(config_file_path, 'r') as file:
        coco_config = json.load(file)
    captions = coco_config['captions']
    for image_id, caption in captions.items():
        reference_caption = caption[0]
        txt_file_name = f"MS_{image_id}.txt"
        txt_file_path = os.path.join(folder_path, txt_file_name)
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                candidate_caption = file.read()
            meteor_score_value = calculate_meteor(reference_caption, candidate_caption)
            meteor_score_values += meteor_score_value
            i += 1
    meteor_score_values /= i
    print(f"METEOR Score: {meteor_score_values}")

def evaluate_rouge(config_file_path, folder_path):
    rouge_score_values = 0
    i = 0
    with open(config_file_path, 'r') as file:
        coco_config = json.load(file)
    captions = coco_config['captions']
    for image_id, caption in captions.items():
        reference_caption = caption[0]
        txt_file_name = f"MS_{image_id}.txt"
        txt_file_path = os.path.join(folder_path, txt_file_name)
        if os.path.exists(txt_file_path):
            with open(txt_file_path, 'r') as file:
                candidate_caption = file.read()
            rouge_score_value = calculate_rouge(reference_caption, candidate_caption)
            rouge_score_values += rouge_score_value
            i += 1
    rouge_score_values /= i
    print(f"ROUGE Score: {rouge_score_values}")

def evaluate_fid(real_images_folder, generated_images_folder):
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], batch_size=64, device=torch.device('cuda'), dims=2048)
    print('FID value:', fid_value)

def evaluate_kid(real_images_folder, generated_images_folder):
    features_generated = nicefid.Features.from_directory(real_images_folder)
    features_real = nicefid.Features.from_directory(generated_images_folder)
    fid = nicefid.compute_fid(features_generated, features_real)
    kid = nicefid.compute_kid(features_generated, features_real)
    print("KID score:", kid)
    print("FID score:", fid)


# Example usage
config_file_path = './MS-COCO_10K.json'
folder_path = '../results/generated_captions/'
real_images_folder = '../results/COCO-10K/'
generated_images_folder = '../results/SDM-MegaFusion-10K/'

evaluate_cider(config_file_path, folder_path)
evaluate_meteor(config_file_path, folder_path)
evaluate_rouge(config_file_path, folder_path)
evaluate_clip_similarity(config_file_path, generated_images_folder)

evaluate_fid(real_images_folder, generated_images_folder)
evaluate_kid(real_images_folder, generated_images_folder)