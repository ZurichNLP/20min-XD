import pandas as pd
#import spacy
import argparse
#from sentence_transformers import SentenceTransformer, util
import os
import torch
from parse_article import get_lead, get_paragraph_list
from multiprocessing import Pool
import numpy as np
import ast
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
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


def process_article(row, nlp_fr, nlp_de):
    fr_sentences = []
    de_sentences = []
    
    # Add titles
    fr_sentences.append({"sentence_id": f'{row["id_fr"]}.1', "aligned_article_id": f'{row["id_de"]}', "sentence_fr": row["head_fr"]})
    de_sentences.append({"sentence_id": f'{row["id_de"]}.1', "aligned_article_id": f'{row["id_fr"]}', "sentence_de": row["head_de"]})

    # Get content
    lead_fr = get_lead(row["content_fr"])[0]
    lead_de = get_lead(row["content_de"])[0]
    content_fr = " ".join(get_paragraph_list(row["content_fr"]))
    content_de = " ".join(get_paragraph_list(row["content_de"]))

    # Split sentences
    fr_sents = nlp_fr(lead_fr + " " + content_fr).sents
    de_sents = nlp_de(lead_de + " " + content_de).sents

    # Add sentences
    for i, sent in enumerate(fr_sents, 2):
        fr_sentences.append({"sentence_id": f'{row["id_fr"]}.{i}', "aligned_article_id": f'{row["id_de"]}', "sentence_fr": sent.text})
    for i, sent in enumerate(de_sents, 2):
        de_sentences.append({"sentence_id": f'{row["id_de"]}.{i}', "aligned_article_id": f'{row["id_fr"]}', "sentence_de": sent.text})

    return fr_sentences, de_sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_aligned_articles", type=str)
    # for sentence splitting it is data/alignments/top15k/model_name.tsv
    # for sentence embedding it is data/split_sentences/model_name/lang/split_sentences.tsv
    # for sentence sim score calculation it is data/embedded_sentences/model_name
    # for sentence alignment it is data/sentence_simscores/model_name.pkl
    parser.add_argument("--split_sentences", action="store_true")
    parser.add_argument("--embed_sentences", action="store_true")
    parser.add_argument("--get_simscores", action="store_true")
    parser.add_argument("--get_sentence_alignments", action="store_true")
    parser.add_argument("--visualize_distribution", action="store_true")
    parser.add_argument("--check_sents", action="store_true")
    args = parser.parse_args()
        

    if args.split_sentences:
        nlp_fr = spacy.load("fr_core_news_sm")
        nlp_de = spacy.load("de_core_news_sm")

        df = pd.read_csv(args.path_to_aligned_articles, encoding="utf-8", sep="\t")

        fr_data = []
        de_data = []

        with Pool() as pool:
            results = pool.starmap(process_article, [(row, nlp_fr, nlp_de) for _, row in df.iterrows()])

        for fr_sents, de_sents in results:
            fr_data.extend(fr_sents)
            de_data.extend(de_sents)

        os.makedirs("data/split_sentences/val-mpnet/fr", exist_ok=True)
        pd.DataFrame(fr_data).to_csv("data/split_sentences/val-mpnet/fr/split_sentences.tsv", index=False, sep="\t")
        os.makedirs("data/split_sentences/val-mpnet/de", exist_ok=True) 
        pd.DataFrame(de_data).to_csv("data/split_sentences/val-mpnet/de/split_sentences.tsv", index=False, sep="\t")

    elif args.embed_sentences:
        if '/de/' in args.path_to_aligned_articles or '_de_' in args.path_to_aligned_articles:
            lang = 'de'
        else:
            lang = 'fr'
        
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        model = SentenceTransformer(model_name, trust_remote_code=True)
        model = model.to(device)

        df = pd.read_csv(args.path_to_aligned_articles, encoding="utf-8", sep="\t")

        embeddings = model.encode(df[f"sentence_{lang}"].tolist(), convert_to_tensor=True, device=device)
        embeddings_numpy = embeddings.cpu().detach().numpy()

        data = df.copy()
        data["embeddings"] = embeddings_numpy.tolist()

        if 'fr' in args.path_to_aligned_articles:
            os.makedirs("data/embedded_sentences/mpnet/fr", exist_ok=True)
            pd.DataFrame(data).to_csv("data/embedded_sentences/mpnet/fr/embedded_sentences.tsv", index=False, sep="\t")
        else:
            os.makedirs("data/embedded_sentences/mpnet/de", exist_ok=True) 
            pd.DataFrame(data).to_csv("data/embedded_sentences/mpnet/de/embedded_sentences.tsv", index=False, sep="\t")

    elif args.get_simscores:
        embeddings_fr = pd.read_csv(os.path.join(args.path_to_aligned_articles, "fr", "embedded_sentences.tsv"), encoding="utf-8", sep="\t")
        embeddings_de = pd.read_csv(os.path.join(args.path_to_aligned_articles, "de", "embedded_sentences.tsv"), encoding="utf-8", sep="\t")

        # get all sentence ids of the french sentences
        ids_fr = embeddings_fr["sentence_id"].tolist()
        ids_fr = [str(id).split(".")[0] for id in ids_fr]
        ids_fr = list(set(ids_fr))

        all_results = []  # List to store all results

        for id_fr in tqdm(ids_fr):
            fr_embs = embeddings_fr[embeddings_fr["sentence_id"].astype(str).str.startswith(id_fr)]["embeddings"].to_list()
            de_embs = embeddings_de[embeddings_de["aligned_article_id"].astype(str) == id_fr]["embeddings"].to_list()
            id_de = str(embeddings_de[embeddings_de["aligned_article_id"].astype(str) == id_fr]["sentence_id"].tolist()[0]).split('.')[0]

            fr_embs = [ast.literal_eval(emb) for emb in fr_embs]
            de_embs = [ast.literal_eval(emb) for emb in de_embs]

            sim_score = util.pytorch_cos_sim(fr_embs, de_embs) * 100

            result = {
                "id_fr": id_fr,
                "id_de": id_de,
                "similarities": sim_score.cpu().numpy().tolist()  # Convert to list instead of numpy array
            }
            all_results.append(result)

        # Save as a regular pickle file instead of npz
        save_path = args.path_to_aligned_articles.replace("embedded_sentences", "sentence_simscores")+'.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Update the corresponding get_sentence_alignments section to use pickle
        with open(save_path, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"Saved simscores to {save_path}")


    elif args.get_sentence_alignments:
        simscores = pickle.load(open(args.path_to_aligned_articles, 'rb'))
        simscores_df = pd.DataFrame(simscores)

        # get sentence ids
        embeddings_fr = pd.read_csv(os.path.join(args.path_to_aligned_articles.replace("sentence_simscores", "embedded_sentences").replace(".pkl", ""), "fr", "embedded_sentences.tsv"), encoding="utf-8", sep="\t")
        sentence_ids_fr = embeddings_fr["sentence_id"].tolist()

        embeddings_de = pd.read_csv(os.path.join(args.path_to_aligned_articles.replace("sentence_simscores", "embedded_sentences").replace(".pkl", ""), "de", "embedded_sentences.tsv"), encoding="utf-8", sep="\t")
        sentence_ids_de = embeddings_de["sentence_id"].tolist()

        # Process each article pair separately
        threshold = 0.0
        all_matches = []
        
        for _, row in simscores_df.iterrows():
            id_fr = row['id_fr']
            id_de = row['id_de']
            similarities = np.array(row['similarities'])
            
            # Get relevant sentence IDs for this article pair
            curr_fr_ids = [sid for sid in sentence_ids_fr if str(sid).startswith(str(id_fr))]
            curr_de_ids = [sid for sid in sentence_ids_de if str(sid).startswith(str(id_de))]
            
            # Find best matches for French sentences
            fr_best_indices = np.where(similarities >= threshold, similarities, -np.inf).argmax(axis=1)
            fr_best_scores = np.max(np.where(similarities >= threshold, similarities, -np.inf), axis=1)
            fr_valid_matches = fr_best_scores > -np.inf
            fr_matches = set(zip([curr_fr_ids[i] for i in range(len(curr_fr_ids)) if fr_valid_matches[i]], 
                               [curr_de_ids[fr_best_indices[i]] for i in range(len(curr_fr_ids)) if fr_valid_matches[i]]))

            # Find best matches for German sentences
            similarities_T = similarities.T
            de_best_indices = np.where(similarities_T >= threshold, similarities_T, -np.inf).argmax(axis=1)
            de_best_scores = np.max(np.where(similarities_T >= threshold, similarities_T, -np.inf), axis=1)
            de_valid_matches = de_best_scores > -np.inf
            de_matches = set(zip([curr_fr_ids[de_best_indices[i]] for i in range(len(curr_de_ids)) if de_valid_matches[i]], 
                               [curr_de_ids[i] for i in range(len(curr_de_ids)) if de_valid_matches[i]]))

            # Keep only matches that are best for both languages
            predicted_matches = fr_matches.intersection(de_matches)
            
            # Get scores for matches
            for fr_id, de_id in predicted_matches:
                fr_idx = curr_fr_ids.index(fr_id)
                de_idx = curr_de_ids.index(de_id)
                score = similarities[fr_idx][de_idx]
                
                # Find existing match if any
                existing_match = next((match for match in all_matches 
                    if match['id_de'] == de_id or match['id_fr'] == fr_id), None)
                
                if existing_match is None:
                    # No existing match found, add new one
                    all_matches.append({
                        'id_de': de_id,
                        'id_fr': fr_id,
                        'score': score
                    })
                elif score > existing_match['score']:
                    # Remove old match and add new higher scoring one
                    all_matches.remove(existing_match)
                    all_matches.append({
                        'id_de': de_id,
                        'id_fr': fr_id, 
                        'score': score
                    })

        df = pd.DataFrame(all_matches)

        # Merge with article data instead of row-by-row operations and drop the embeddings columns
        df = df.merge(
            embeddings_fr,
            left_on='id_fr',
            right_on='sentence_id',
            how='left'  
        ).merge(
            embeddings_de,
            left_on='id_de',
            right_on='sentence_id',
            how='left',
            suffixes=('_fr', '_de')
        ).drop(columns=['embeddings_fr', 'embeddings_de'])

        # deduplicate rows before saving
        df = df.drop_duplicates(subset=['id_de', 'id_fr'])
        os.makedirs(os.path.dirname(args.path_to_aligned_articles.replace("sentence_simscores", "sentence_alignments")), exist_ok=True)
        df.to_csv(args.path_to_aligned_articles.replace("sentence_simscores", "sentence_alignments").replace('.pkl', '.tsv'), sep="\t", index=False)    
        
    elif args.visualize_distribution:
        simscores_df = pd.read_csv(args.path_to_aligned_articles, encoding="utf-8", sep="\t")

        # only keep rows with score > 46.0
        simscores_df = simscores_df[simscores_df['score'] > 46.0]
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(simscores_df['score'], simscores_df['char_count_diff'], 
                   alpha=0.1, s=1, color='#ADD8E6')
        
        # Add mean lines
        plt.axvline(x=78.64999999, color='grey', linestyle='--', linewidth=0.8, 
                   label=f'Top 15k cut: 78.64')
        plt.axvline(x=46.0, color='grey', linestyle=':', linewidth=0.8,
                   label=f'Threshold: 46.0')
        
        # Add smoothed trend line
        scores_sorted = sorted(simscores_df['score'])
        z = np.polyfit(simscores_df['score'], simscores_df['char_count_diff'], 1)
        p = np.poly1d(z)
        plt.plot(scores_sorted, p(scores_sorted), "r--", alpha=0.8,
                label='Trend line')
        
        # Set labels and title
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Characters')
        plt.title('Sentence Similarity vs LengthDiff')
        
        # Set x-axis minimum to 0
        plt.xlim(left=0)
        
        # Add legend with increased opacity
        plt.legend(framealpha=1.0, loc='upper right')
        
        plt.show()

        simscores_df = pd.read_csv(args.path_to_aligned_articles, encoding="utf-8", sep="\t")
        
        # Create histogram plot with same size
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        n, bins, patches = plt.hist(simscores_df['score'], bins=100, density=False, color='#ADD8E6', alpha=0.7)
        
        # Add mean line
        mean_score = simscores_df['score'].mean()
        plt.axvline(x=mean_score, color='grey', linestyle='--', linewidth=0.8,
                   label=f'Mean: {mean_score:.2f}')
        
        # Set labels and title
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Sentence Similarity Score Distribution')
        
        # Set x-axis minimum to 0
        plt.xlim(left=46.)
        
        # Add legend
        plt.legend()
        
        
        plt.show()


    elif args.check_sents:
        # check if the sentences are aligned
        aligned_articles = pd.read_csv(args.path_to_aligned_articles, encoding="utf-8", sep="\t")
        # print all that have kendall_tau > 0.9
        print(aligned_articles[aligned_articles['score'] > 46.0])

        
        """# get number of aligned sentences per article pair
        aligned_sentences = pd.read_csv('data/sentence_alignments/mpnet_highest_match_non_sents_removed_sentence_length.tsv', encoding="utf-8", sep="\t")
        # get number of aligned sentences per article pair
        aligned_sentences = aligned_sentences.groupby(['aligned_article_id_de', 'aligned_article_id_fr']).size().reset_index(name='num_aligned_sentences')
        print(aligned_sentences)

        # print all that have num_aligned_sentences > 1
        print(aligned_sentences[aligned_sentences['num_aligned_sentences'] > 1])
        """

        

    


if __name__ == "__main__":
    main()