import numpy as np
from numpy.linalg import norm
import pandas as pd
import os
import tqdm
import argparse
from sentence_transformers import util

# ids of manually selected matching articles
MATCH_FR = [54531155, 54533662, 54533882, 54533659, 54533363, 54532039, 54529607, 54534371, 54534307, 54531759, 54525986, 54531758, 54533881, 54532036]
MATCH_DE = [54529608, 54529609, 54531762, 54531164, 54532032, 54533666, 54529610, 54530322, 54524974, 54531464, 54532033, 54531160, 54533663, 54534158]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_val_simscores", type=str)
    parser.add_argument("align_type", type=str, default="above_threshold")
    args = parser.parse_args()

    simscores = np.load(args.path_to_val_simscores)

    similarities = simscores["similarities"]
    ids_fr = simscores["ids_fr"]
    ids_de = simscores["ids_de"]

    assert similarities.shape == (len(ids_fr), len(ids_de))

    best_f1 = 0
    best_threshold = 0
    thresholds = np.arange(0, 100, 0.5) 
    true_matches = set((fr_id, de_id) for fr_id, de_id in zip(MATCH_FR, MATCH_DE))

    for threshold in thresholds:
        # go through all scores in the matrix and add all matches above threshold to predicted matches
        if args.align_type == "above_threshold":
            predicted_matches = set()
            for i in range(len(similarities)):
                for j in range(len(similarities[i])):
                    if similarities[i][j] >= threshold:
                        predicted_matches.add((ids_fr[i], ids_de[j]))
        
        # get predicted matches above threshold and best match for each French article
        elif args.align_type == "best_fr":
            predicted_matches = set()
            fr_articles = {}
            for i in range(len(similarities)):
                for j in range(len(similarities[i])):
                    if similarities[i][j] >= threshold:
                        if ids_fr[i] not in fr_articles or similarities[i][j] > fr_articles[ids_fr[i]][0]:
                            fr_articles[ids_fr[i]] = (similarities[i][j], ids_de[j])
            predicted_matches = set((fr_id, de_id) for fr_id, (score, de_id) in fr_articles.items())
            
        # get predicted matches above threshold and best match for each German article
        elif args.align_type == "best_de":
            predicted_matches = set()
            de_articles = {}
            for i in range(len(similarities)):
                for j in range(len(similarities[i])):
                    if similarities[i][j] >= threshold:
                        if ids_de[j] not in de_articles or similarities[i][j] > de_articles[ids_de[j]][0]:
                            de_articles[ids_de[j]] = (similarities[i][j], ids_fr[i])
            predicted_matches = set((fr_id, de_id) for de_id, (score, fr_id) in de_articles.items())
        
        # get predicted matches above threshold that are the best for each article no matter the language
        elif args.align_type == "best_both":
            predicted_matches = set()
            fr_articles = {}
            de_articles = {}
            for i in range(len(similarities)):
                for j in range(len(similarities[i])):
                    if similarities[i][j] >= threshold:
                        if ids_fr[i] not in fr_articles or similarities[i][j] > fr_articles[ids_fr[i]][0]:
                            fr_articles[ids_fr[i]] = (similarities[i][j], ids_de[j])
                        if ids_de[j] not in de_articles or similarities[i][j] > de_articles[ids_de[j]][0]:
                            de_articles[ids_de[j]] = (similarities[i][j], ids_fr[i])
            predicted_matches = set((fr_id, de_id) for fr_id, (score, de_id) in fr_articles.items())
            predicted_matches.update(set((fr_id, de_id) for de_id, (score, fr_id) in de_articles.items()))
        
        # get predicted matches above threshold that are the best only if they are the best for both articles compared to all other articles
        elif args.align_type == "best":
            predicted_matches = set()
            fr_articles = {}
            de_articles = {}
            for i in range(len(similarities)):
                for j in range(len(similarities[i])):
                    if similarities[i][j] >= threshold:
                        if ids_fr[i] not in fr_articles or similarities[i][j] > fr_articles[ids_fr[i]][0]:
                            fr_articles[ids_fr[i]] = (similarities[i][j], ids_de[j])
                        if ids_de[j] not in de_articles or similarities[i][j] > de_articles[ids_de[j]][0]:
                            de_articles[ids_de[j]] = (similarities[i][j], ids_fr[i])        
            predicted_matches = set((fr_id, de_id) for fr_id, (score, de_id) in fr_articles.items())
            predicted_matches = predicted_matches.intersection(set((fr_id, de_id) for de_id, (score, fr_id) in de_articles.items()))

        # calculate precision and recall
        true_positives = len(predicted_matches.intersection(true_matches))
        precision = true_positives / len(predicted_matches) if predicted_matches else 0
        recall = true_positives / len(true_matches) if true_matches else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"\nAlign type: {args.align_type}")
    print(f"Model: {args.path_to_val_simscores.split('_')[-1].split('.')[0]}")
    print(f"Threshold: {best_threshold}, F1: {best_f1}")

    # Save results to file
    with open('threshold_results.txt', 'a') as f:
        f.write(f"\nAlign type: {args.align_type}\n")
        f.write(f"Model: {args.path_to_val_simscores.split('_')[-1].split('.')[0]}\n")
        f.write(f"Threshold: {best_threshold}, F1: {best_f1}\n")

if __name__ == "__main__":
    main()

