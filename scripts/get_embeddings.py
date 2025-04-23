import numpy as np
import parse_article
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import torch
import argparse
from sentence_transformers import util


from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("path_to_articles", type=str)
parser.add_argument("full_hf_model_name", type=str)
args = parser.parse_args()

# set environment variable for cached models and files by huggingface
os.environ["HF_HOME"] = ".hf/"


# Check available CUDA GPUs and setup multi-GPU
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    print(f"Number of CUDA GPUs available: {n_gpus}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders) device for MacBook")
else:
    device = torch.device("cpu")
    print("Using CPU - no GPU detected")


def generate_sentence_embeddings(texts, tokenizer, model):
    embeddings = []
    model = model.to(device)  # Ensure model is on the correct device
    
    for text in texts:
        # Move inputs to the same device as model
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Perform mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings.append(torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9))
    
    # Stack all embeddings and return as numpy array
    return torch.cat(embeddings, dim=0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_articles", type=str)
    parser.add_argument("full_hf_model_name", type=str)
    args = parser.parse_args()

    articles = pd.read_csv(args.path_to_articles, 
                            sep="\t", 
                            encoding='utf-8')

    # sort by pubdate
    articles = articles.sort_values(by='pubdate')


    if '/de/' in args.path_to_articles or '_de_' in args.path_to_articles:
        lang = 'de'
    else:
        lang = 'fr'


    print(f"\nLanguage: {lang}")
    print(f"Path to articles: {args.path_to_articles}")
    print(f"Number of articles: {len(articles)}")

    # load model
    model_name = args.full_hf_model_name
    # sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    # Alibaba-NLP/gte-modernbert-base
    # sentence-transformers/LaBSE
    # Alibaba-NLP/gte-multilingual-base
    # Parallia/Fairly-Multilingual-ModernBERT-Embed-BE

    if 'swissbert' in model_name:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.set_default_language('de_CH') if 'de' in args.path_to_articles else model.set_default_language('fr_CH')

    else:   
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model = model.to(device)


    for day in articles['pubdate'].unique():
        # Check if any file in the embeddings directory contains this day's date
        os.makedirs(f'data/embeddings/{model_name.split("/")[-1]}/{lang}', exist_ok=True)
        if any(day in filename for filename in os.listdir(f'data/embeddings/{model_name.split("/")[-1]}/{lang}')):
            print(f"Skipping {day} because it already has embeddings")
            
        else:
            texts = []
            ids = []

            articles_day = articles[articles['pubdate'] == day]
            
            for _, art in articles_day.iterrows():
                try:
                    title = art['head']
                    lead = parse_article.get_lead(art['content'])[0]
                    text = title + " " + lead
                    texts.append(text)
                    ids.append(art['content_id']) if 'content_id' in art else ids.append(art['id'])
                except:
                    continue
            
            if 'swissbert' in model_name:
                try:
                    embeddings_numpy = generate_sentence_embeddings(texts, tokenizer, model)
                except:
                    print(f"There was an error with {day}.")
                    continue
            else:
                embeddings = model.encode(texts, convert_to_tensor=True, device=device)
                embeddings_numpy = embeddings.cpu().detach().numpy()
            
            # save embeddings
            if 'test' in args.path_to_articles:
                os.makedirs(f'data/val_embeddings/{lang}', exist_ok=True)
                np.savez(f'data/val_embeddings/{lang}/{day}_{model_name.split("/")[-1]}.npz', embeddings=embeddings_numpy, ids=ids)
                print(f'Saved embeddings for {day} to {lang}/val_embeddings/{day}_{model_name.split("/")[-1]}.npz')
            else:
                os.makedirs(f'data/embeddings/{model_name.split("/")[-1]}/{lang}', exist_ok=True)
                np.savez(f'data/embeddings/{model_name.split("/")[-1]}/{lang}/{day}_{model_name.split("/")[-1]}.npz', embeddings=embeddings_numpy, ids=ids)
                print(f'Saved embeddings for {day} to {lang}/embeddings/{day}_{model_name.split("/")[-1]}.npz')


if __name__ == "__main__":
    main()

    