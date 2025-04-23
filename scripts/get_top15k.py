import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_articles", type=str, default='data/alignments/mpnet')
    args = parser.parse_args()

    # Get all file paths first
    file_paths = [os.path.join(args.path_to_articles, f) for f in os.listdir(args.path_to_articles)]
    
    # Read all files at once using list comprehension
    dfs = [pd.read_csv(f, sep='\t', encoding='utf-8', on_bad_lines='skip') for f in file_paths]
    
    # Concatenate all dataframes in one operation
    df = pd.concat(dfs, ignore_index=True)

    df = df.sort_values(by='score', ascending=False)

    print("\nBEFORE FILTERING")
    #print(df[['score', 'id_de', 'id_fr', 'head_de', 'head_fr']].head(10))
    print(f'Number of article pairs: {len(df)}')
    print(f'Min similarity score: {df["score"].min()}')
    print(f'Max similarity score: {df["score"].max()}')

    # drop articles that are faulty
    df = df[~df['head_de'].str.startswith('Dieser Inhalt funktioniert nur')]
    # drop articles that have the same text in both languages
    df = df[df['head_de'].str[:10] != df['head_fr'].str[:10]]

    # and show scores as well as ids and headlines
    print("\nAFTER FILTERING")
    #print(df[['score', 'id_de', 'id_fr', 'head_de', 'head_fr']].head(10))
    print(f'Number of article pairs after filtering: {len(df)}')
    print(f'Min similarity score: {df["score"].min()}')
    print(f'Max similarity score: {df["score"].max()}')

    # save the filtered dataframe
    os.makedirs(args.path_to_articles+'_full', exist_ok=True)
    df.to_csv(f'{args.path_to_articles}_full/{args.path_to_articles.split("/")[-1]}.tsv', index=False, sep='\t', encoding='utf-8')
    print(f'\nSaved to {args.path_to_articles}_full/{args.path_to_articles.split("/")[-1]}.tsv')

    df = df.head(15000)
    print("\nTOP 15K ARTICLE PAIRS BY SIMILARITY SCORE")
    print(f'Number of article pairs: {len(df)}')
    print(f'Min similarity score: {df["score"].min()}')
    print(f'Max similarity score: {df["score"].max()}')

    print("top 10")
    print(df[['score', 'id_de', 'id_fr', 'head_de', 'head_fr']].head(10))
    print("bottom 10")
    print(df[['score', 'id_de', 'id_fr', 'head_de', 'head_fr']].tail(10))


    # save the filtered dataframe
    os.makedirs('data/alignments/top15k', exist_ok=True)
    df.to_csv(f'data/alignments/top15k/{args.path_to_articles.split("/")[-1]}.tsv', index=False, sep='\t', encoding='utf-8')
    print(f'\nSaved to data/alignments/top15k/{args.path_to_articles.split("/")[-1]}.tsv')

if __name__ == "__main__":
    main()