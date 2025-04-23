import numpy as np
from numpy.linalg import norm
import os
from concurrent.futures import ProcessPoolExecutor
import tqdm
import argparse
from sentence_transformers import util
import pandas as pd
import time

def process_single_file(file_data):
    file_path, args = file_data
    simscores = np.load(file_path)
    similarities = simscores["similarities"]
    ids_fr = simscores["ids_fr"]
    ids_de = simscores["ids_de"]
    threshold = 46

    # Vectorized operations for finding matches
    if args.align_type == "above_threshold":
        # Use numpy where to find all matches above threshold
        matches = np.where(similarities >= threshold)
        predicted_matches = list(zip(ids_fr[matches[0]], ids_de[matches[1]]))
        scores_of_matches = similarities[matches]

    elif args.align_type == "best_fr":
        # Use numpy operations to find best matches
        best_indices = np.where(similarities >= threshold, similarities, -np.inf).argmax(axis=1)
        best_scores = np.max(np.where(similarities >= threshold, similarities, -np.inf), axis=1)
        valid_matches = best_scores > -np.inf
        predicted_matches = list(zip(ids_fr[valid_matches], ids_de[best_indices[valid_matches]]))
        scores_of_matches = best_scores[valid_matches]
        
    elif args.align_type == "best_de":
        # Use numpy operations to find best matches
        best_indices = np.where(similarities.T >= threshold, similarities.T, -np.inf).argmax(axis=1)
        best_scores = np.max(np.where(similarities.T >= threshold, similarities.T, -np.inf), axis=1)
        valid_matches = best_scores > -np.inf
        predicted_matches = list(zip(ids_fr[best_indices[valid_matches]], ids_de[valid_matches]))
        scores_of_matches = best_scores[valid_matches]

    elif args.align_type == "best_both":
        # Find best matches for French articles
        fr_best_indices = np.where(similarities >= threshold, similarities, -np.inf).argmax(axis=1)
        fr_best_scores = np.max(np.where(similarities >= threshold, similarities, -np.inf), axis=1)
        fr_valid_matches = fr_best_scores > -np.inf
        fr_matches = set(zip(ids_fr[fr_valid_matches], ids_de[fr_best_indices[fr_valid_matches]]))

        # Find best matches for German articles
        de_best_indices = np.where(similarities.T >= threshold, similarities.T, -np.inf).argmax(axis=1)
        de_best_scores = np.max(np.where(similarities.T >= threshold, similarities.T, -np.inf), axis=1)
        de_valid_matches = de_best_scores > -np.inf
        de_matches = set(zip(ids_fr[de_best_indices[de_valid_matches]], ids_de[de_valid_matches]))

        # Combine matches
        predicted_matches = list(fr_matches.union(de_matches))
        scores_of_matches = [similarities[ids_fr.tolist().index(fr_id)][ids_de.tolist().index(de_id)] for fr_id, de_id in predicted_matches]

    elif args.align_type == "best":
        # Find best matches for French articles
        fr_best_indices = np.where(similarities >= threshold, similarities, -np.inf).argmax(axis=1)
        fr_best_scores = np.max(np.where(similarities >= threshold, similarities, -np.inf), axis=1)
        fr_valid_matches = fr_best_scores > -np.inf
        fr_matches = set(zip(ids_fr[fr_valid_matches], ids_de[fr_best_indices[fr_valid_matches]]))

        # Find best matches for German articles
        de_best_indices = np.where(similarities.T >= threshold, similarities.T, -np.inf).argmax(axis=1)
        de_best_scores = np.max(np.where(similarities.T >= threshold, similarities.T, -np.inf), axis=1)
        de_valid_matches = de_best_scores > -np.inf
        de_matches = set(zip(ids_fr[de_best_indices[de_valid_matches]], ids_de[de_valid_matches]))

        # Keep only matches that are best for both languages
        predicted_matches = list(fr_matches.intersection(de_matches))
        scores_of_matches = [similarities[ids_fr.tolist().index(fr_id)][ids_de.tolist().index(de_id)] for fr_id, de_id in predicted_matches]

    # Read articles data only once and store in memory
    if not hasattr(process_single_file, 'fr_article'):
        process_single_file.fr_article = pd.read_csv(f"{args.path_to_articles}/fr/articles_cleaned_dates.tsv", sep="\t")
        process_single_file.de_article = pd.read_csv(f"{args.path_to_articles}/de/articles_cleaned_dates.tsv", sep="\t")

    # Create DataFrame more efficiently
    matches_data = {
        'id_de': [de_id for _, de_id in predicted_matches],
        'id_fr': [fr_id for fr_id, _ in predicted_matches],
        'score': scores_of_matches
    }
    df = pd.DataFrame(matches_data)

    # Merge with article data instead of row-by-row operations
    df = df.merge(
        process_single_file.de_article,
        left_on='id_de',
        right_on='content_id',
        how='left'
    ).merge(
        process_single_file.fr_article,
        left_on='id_fr',
        right_on='content_id',
        how='left',
        suffixes=('_de', '_fr')
    )

    # deduplicate rows before saving
    df = df.drop_duplicates(subset=['id_de', 'id_fr'])
    os.makedirs(os.path.dirname(file_path.replace('simscores/', 'alignments/')), exist_ok=True)
    df.to_csv(file_path.replace('simscores/', 'alignments/').replace('.npz', '.tsv'), sep="\t", index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_simscores", type=str)
    parser.add_argument("path_to_articles", type=str)
    parser.add_argument("align_type", type=str, default="best")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    files = [(os.path.join(args.path_to_simscores, f), args) 
             for f in os.listdir(args.path_to_simscores)]

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm.tqdm(executor.map(process_single_file, files), total=len(files)))

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    print(f"Script execution time: {elapsed_time:.2f} seconds")