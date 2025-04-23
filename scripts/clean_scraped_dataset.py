import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path_to_articles", type=str, default='data/scraped_articles/fr/articles.tsv')
parser.add_argument("--clean-articles", type=bool, default=False)
parser.add_argument("--drop-dates", type=bool, default=False)
parser.add_argument("--split-dates", type=bool, default=False)
args = parser.parse_args()


def clean_articles(df):
    print(f"df columns: {df.columns}")
    print(f'df.info(): {df.info()}')

    # print df size before deduplication
    print(f"df size before dropping rows w/o content: {len(df)}")

    df = df.dropna(subset=['content'])

    # print df size after dropping rows w/o content
    print(f"df size after dropping rows w/o content: {len(df)}")    

    # deduplicate articles
    df.drop_duplicates(subset=['article_link'], inplace=True)

    # print df size after deduplication
    print(f"df size after deduplication: {len(df)}")

    # remove video articles
    df = df[~df['article_link'].str.contains('video')]
    df = df[~df['article_link'].str.contains('vidéo')]

    # print df size after removing video articles
    print(f"df size after removing video articles: {len(df)}")

    # saving cleaned dataset
    df.to_csv('data/scraped_articles/fr/articles_cleaned.tsv', sep='\t', index=False)


def drop_dates():
    df_fr = pd.read_csv('data/scraped_articles/fr/articles_cleaned.tsv', sep='\t', encoding='utf-8', on_bad_lines='skip')
    df_de = pd.read_csv('data/scraped_articles/de/articles_cleaned.tsv', sep='\t', encoding='utf-8', on_bad_lines='skip')

    # print df size before dropping dates
    print(f"df_fr size before dropping dates: {len(df_fr)}")
    print(f"df_de size before dropping dates: {len(df_de)}")

    # keep only dates that are in both dfs
    df_fr = df_fr[df_fr['pubdate'].isin(df_de['pubdate'])]
    df_de = df_de[df_de['pubdate'].isin(df_fr['pubdate'])]

    # print df size after dropping dates
    print(f"df_fr size after dropping dates: {len(df_fr)}")
    print(f"df_de size after dropping dates: {len(df_de)}")

    # saving cleaned dataset
    df_fr.to_csv('data/scraped_articles/fr/articles_cleaned_dates.tsv', sep='\t', index=False)
    df_de.to_csv('data/scraped_articles/de/articles_cleaned_dates.tsv', sep='\t', index=False)


# this is only to parallelize further processing
def split_dates(df):
    # Get unique dates first and then split
    df = df.sort_values(by='pubdate')
    unique_dates = df['pubdate'].unique()
    mid_point = len(unique_dates)//2

    with open('data/half_dates.txt', 'w') as f, open('data/half_dates2.txt', 'w') as f2:
        for date in unique_dates[:mid_point]:
            f.write(f"{date}\n")
        for date in unique_dates[mid_point:]:
            f2.write(f"{date}\n")


def main():
    df = pd.read_csv(args.path_to_articles, sep='\t', encoding='utf-8', on_bad_lines='skip')
    
    if args.clean_articles:
        clean_articles(df)
    if args.drop_dates:
        drop_dates()
    if args.split_dates:
        split_dates(df)


if __name__ == "__main__":
    main()

