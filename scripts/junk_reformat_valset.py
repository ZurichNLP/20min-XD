import pandas as pd

def main():
    df = pd.read_csv("data/test/20min_align_28aug.tsv", sep="\t")

    df_new = pd.DataFrame(columns=["id_de",	"id_fr",	"content_id_de",	"pubtime_de",	"article_link_de",	"medium_code_de",	"medium_name_de",	"language_de",	"char_count_de",	"head_de",	"subhead_de",	"content_de",	"content_id_fr",	"pubtime_fr",	"article_link_fr",	"medium_code_fr",	"medium_name_fr",	"language_fr",	"char_count_fr",	"head_fr",	"subhead_fr", "content_fr"])

    df_fr = df[df['language'] == 'fr']
    df_de = df[df['language'] == 'de']

    for (index_fr, row_fr), (index_de, row_de) in zip(df_fr.iterrows(), df_de.iterrows()):
        df_new.loc[index_fr] = [row_de['id'], row_fr['id'], row_de['content_id'], row_de['pubtime'], row_de['article_link'], row_de['medium_code'], row_de['medium_name'], row_de['language'], row_de['char_count'], row_de['head'], row_de['subhead'], row_de['content'], row_fr['content_id'], row_fr['pubtime'], row_fr['article_link'], row_fr['medium_code'], row_fr['medium_name'], row_fr['language'], row_fr['char_count'], row_fr['head'], row_fr['subhead'], row_fr['content']]

    df_new.to_csv("data/test/val_set.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()