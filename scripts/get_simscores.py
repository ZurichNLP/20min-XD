import numpy as np
from numpy.linalg import norm
import os
import tqdm
import argparse
from sentence_transformers import util


def load_and_get_simscores(filename_fr, filename_de):
    embeddings_fr = np.load(filename_fr)
    embeddings_de = np.load(filename_de)

    ids_fr = embeddings_fr["ids"]
    ids_de = embeddings_de["ids"]

    assert len(ids_de) == len(embeddings_de["embeddings"]) and len(ids_fr) == len(embeddings_fr["embeddings"])

    similarities = util.pytorch_cos_sim(embeddings_fr["embeddings"], embeddings_de["embeddings"]) * 100

    assert similarities.shape == (len(ids_fr), len(ids_de))

    save_path = filename_fr.replace("embeddings", "simscores").replace('/fr', '')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, similarities=similarities, ids_fr=ids_fr, ids_de=ids_de)
    return save_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_embeddings", type=str)
    parser.add_argument("full_hf_model_name", type=str)
    parser.add_argument("--pubdate", type=str)
    args = parser.parse_args()

    if not args.pubdate:
        for filename_de, filename_fr in tqdm.tqdm(zip(os.listdir(args.path_to_embeddings+"/de/"), os.listdir(args.path_to_embeddings+"/fr/"))):
            print(f"Processing {filename_de} and {filename_fr}")
            if args.full_hf_model_name.split("/")[-1] in filename_de and args.full_hf_model_name.split("/")[-1] in filename_fr:
                
                day_de = filename_de.split("_")[0]
                day_fr = filename_fr.split("_")[0]

                assert day_de == day_fr

                model_name_de = filename_de.split("_")[1]
                model_name_fr = filename_fr.split("_")[1]

                assert model_name_de == model_name_fr

                try:
                    save_path = load_and_get_simscores(args.path_to_embeddings+"/fr/"+filename_fr, args.path_to_embeddings+"/de/"+filename_de)
                    print(f"Saved simscores to {save_path}")
                except:
                    print(f"There was an error with {filename_fr} and {filename_de}.")
                    continue
    else:       
        save_path = load_and_get_simscores(args.path_to_embeddings+"/fr/"+args.pubdate+"_"+args.full_hf_model_name.split("/")[-1]+".npz", 
                                         args.path_to_embeddings+"/de/"+args.pubdate+"_"+args.full_hf_model_name.split("/")[-1]+".npz")
        print(f"Saved simscores to {save_path}")

if __name__ == "__main__":
    main()

