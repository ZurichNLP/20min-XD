import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_aligned_sentences", type=str)
    # data/sentence_alignments/mpnet.tsv
    parser.add_argument("--get_highest_match", action="store_true")
    parser.add_argument("--remove_non_sents", action="store_true")
    parser.add_argument("--get_sentence_length", action="store_true")
    args = parser.parse_args()

    aligned_sentences = pd.read_csv(args.path_to_aligned_sentences, encoding="utf-8", sep="\t")

    if args.get_highest_match:
        print(f"There are {len(aligned_sentences)} rows in the aligned sentences dataframe.")

        # Sort by score and drop duplicates keeping first (highest scoring) occurrence
        new_df = aligned_sentences.sort_values('score', ascending=False)\
                                .drop_duplicates(subset=['id_de'], keep='first')\
                                .drop_duplicates(subset=['id_fr'], keep='first')

        print(f"There are {len(new_df)} rows in the new dataframe.")

        new_df.to_csv(args.path_to_aligned_sentences.replace(".tsv", "_highest_match.tsv"), sep="\t", index=False)

    if args.remove_non_sents:
        print(f"There are {len(aligned_sentences)} rows in the aligned sentences dataframe.")

        # Define patterns to filter out
        patterns_to_remove = [
            '(pat/sda)', 'nxp/afp)', '(', ')', '«', '»',
            '(rca/20 minutes/ats)', '(nxp/ats)', '(nxp/afp)', '-', 'ats', '(rar/sda)',
            'ats', '» «', '(nxp/20 minutes/afp)', 

        ]

        # Create mask for filtering patterns and short sentences
        mask = ~(
            aligned_sentences['sentence_de'].isin(patterns_to_remove) |
            aligned_sentences['sentence_fr'].isin(patterns_to_remove)
        ) & (
            aligned_sentences['sentence_de'].str.len() > 29
        ) & (
            aligned_sentences['sentence_fr'].str.len() > 29
        )

        # Filter dataframe
        new_df = aligned_sentences[mask]

        # sort by id_de
        new_df = new_df.sort_values('id_de')

        print(f"There are {len(new_df)} rows in the new dataframe.")

        new_df.to_csv(args.path_to_aligned_sentences.replace(".tsv", "_non_sents_removed.tsv"), sep="\t", index=False)
    
    if args.get_sentence_length:
        new_df = aligned_sentences
        # add columns char_count_de and char_count_fr
        new_df['char_count_de'] = new_df['sentence_de'].str.len()
        new_df['char_count_fr'] = new_df['sentence_fr'].str.len()
        new_df['char_count_diff'] = abs(new_df['char_count_de'] - new_df['char_count_fr'])

        # save to csv
        new_df.to_csv(args.path_to_aligned_sentences.replace(".tsv", "_sentence_length.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()