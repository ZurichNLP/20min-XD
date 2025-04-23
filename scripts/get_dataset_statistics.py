import pandas as pd
import os
import numpy as np
import argparse
from parse_article import get_lead, get_paragraph_list
import spacy
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from scipy import stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=str, help="Path to the dataset")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer to use", default="mpnet")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--extremes", action="store_true")
    parser.add_argument("--correlation_test", action="store_true")
    parser.add_argument("--num_aligned_sentences", action="store_true")
    parser.add_argument("--vis_align", action="store_true")
    parser.add_argument("--mono", action="store_true")
    parser.add_argument("--length_correlation", action="store_true")
    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Load the dataset
    dataset = pd.read_csv(dataset_path, sep="\t")
    #print(dataset.head())

    if not args.viz and not args.extremes and not args.correlation_test and not args.num_aligned_sentences and not args.vis_align and not args.mono and not args.length_correlation:
        nlp_fr = spacy.load("fr_core_news_sm")
        nlp_de = spacy.load("de_core_news_sm")
        
        # Print the number of rows in the dataset
        print(f"Number of rows in the dataset: {len(dataset)}")

        # Print avg number of chars in head
        print(f"Average number of chars in head GERMAN: {np.mean([len(x) for x in dataset['head_de']])}")
        print(f"Average number of chars in head FRENCH: {np.mean([len(x) for x in dataset['head_fr']])}")

        # Print avg number of chars in all leads  
        leads_de = [get_lead(x) for x in dataset['content_de']]
        leads_fr = [get_lead(x) for x in dataset['content_fr']]
        #leads_de = [x.split("<ld>")[1].split("</ld>")[0].replace('&amp;', '').replace('amp;', '').replace('lt;', '').replace('gt;', '').replace('tx', '').replace('ld', '').replace('&<&>&<&>&<p&>', '') for x in dataset['content_de'] if "<ld>" in x]
        #leads_fr = [x.split("<ld>")[1].split("</ld>")[0].replace('&amp;', '').replace('amp;', '').replace('lt;', '').replace('gt;', '').replace('tx', '').replace('ld', '').replace('&<&>&<&>&<p&>', '') for x in dataset['content_fr'] if "<ld>" in x]
        # leads is a list of lists of strings
        print(f"Average number of chars in lead GERMAN: {np.mean([len(x[0]) for x in leads_de])}") # TODO:this currently does not work
        print(f"Average number of chars in lead FRENCH: {np.mean([len(x[0]) for x in leads_fr])}") # TODO:this currently does not work
        sum_leads_chars = sum([len(x) for x in leads_de]) + sum([len(x) for x in leads_fr])

        # Print avg number of chars in content
        contents_de = [get_paragraph_list(x) for x in dataset['content_de']]
        contents_fr = [get_paragraph_list(x) for x in dataset['content_fr']]

        #contents_de = [x.split("</ld>")[1].replace('&amp;', '').replace('amp;', '').replace('lt;', '').replace('gt;', '').replace('tx', '').replace('ld', '').replace('&<&>&<&>&<p&>', '') if '&lt;ld&gt;' in x else x.replace('&amp;', '').replace('amp;', '').replace('lt;', '').replace('gt;', '').replace('tx', '').replace('ld', '').replace('&<&>&<&>&<p&>', '') for x in dataset['content_de']]
        #contents_fr = [x.split("</ld>")[1].replace('&amp;', '').replace('amp;', '').replace('lt;', '').replace('gt;', '').replace('tx', '').replace('ld', '').replace('&<&>&<&>&<p&>', '') if '&lt;ld&gt;' in x else x.replace('&amp;', '').replace('amp;', '').replace('lt;', '').replace('gt;', '').replace('tx', '').replace('ld', '').replace('&<&>&<&>&<p&>', '') for x in dataset['content_fr']]
        print(f"Average number of chars in content GERMAN: {np.mean([len(''.join(x)) for x in contents_de])}")
        print(f"Average number of chars in content FRENCH: {np.mean([len(''.join(x)) for x in contents_fr])}")
        sum_content_chars = sum([len(x) for x in contents_de]) + sum([len(x) for x in contents_fr])

        # Print avg number of sentences per article
        sum_sentences_de = 0
        sum_sentences_fr = 0
        for article in contents_de:
            doc = nlp_de(" ".join(article))
            sum_sentences_de += len(list(doc.sents))
        for article in contents_fr:
            doc = nlp_fr(" ".join(article))
            sum_sentences_fr += len(list(doc.sents))
        print(f"Average number of sentences in article GERMAN: {sum_sentences_de/len(contents_de)}")
        print(f"Average number of sentences in article FRENCH: {sum_sentences_fr/len(contents_fr)}")
        #print(f"Average number of sentences in paragraphs: {np.mean([len(x) for x in sentences])}")

        # print total number of chars in leads and content and heads
        print(f"Total number of chars in content for GERMAN: {sum([len(x) for x in dataset['head_de']])+sum([len(x[0]) for x in leads_de])+sum([len(''.join(x)) for x in contents_de])}")
        print(f"Total number of chars in content for FRENCH: {sum([len(x) for x in dataset['head_fr']])+sum([len(x[0]) for x in leads_fr])+sum([len(''.join(x)) for x in contents_fr])}")

        # get and print total number of tokens in in full text using huggingface tokenizer mpnet 
        if args.tokenizer == "mpnet":
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        elif args.tokenizer == "bert":
            tokenizer = AutoTokenizer.from_pretrained("jgrosjean-mathesis/sentence-swissbert")

        title_tokens_de = tokenizer([x for x in dataset['head_de']], padding=False)
        lead_tokens_de = tokenizer([x[0] for x in leads_de], padding=False)
        total_content_tokens_de = 0
        for x in contents_de:
            para_tokens_de = tokenizer(x, padding=False)
            total_content_tokens_de += sum(len(ids) for ids in para_tokens_de['input_ids'])
        
        title_tokens_fr = tokenizer([x for x in dataset['head_fr']], padding=False)
        lead_tokens_fr = tokenizer([x[0] for x in leads_fr], padding=False)
        total_content_tokens_fr = 0
        for x in contents_fr:
            para_tokens_fr = tokenizer(x, padding=False)
            total_content_tokens_fr += sum(len(ids) for ids in para_tokens_fr['input_ids'])
        
        print(f"Average number of tokens in head for GERMAN: {np.mean([len(ids) for ids in title_tokens_de['input_ids']])}")
        print(f"Average number of tokens in head for FRENCH: {np.mean([len(ids) for ids in title_tokens_fr['input_ids']])}")

        print(f"Average number of tokens in lead for GERMAN: {np.mean([len(ids) for ids in lead_tokens_de['input_ids']])}")
        print(f"Average number of tokens in lead for FRENCH: {np.mean([len(ids) for ids in lead_tokens_fr['input_ids']])}")

        print(f"Total number of tokens in content for GERMAN: {total_content_tokens_de}")
        print(f"Total number of tokens in content for FRENCH: {total_content_tokens_fr}")

        print(f"Total number of tokens in head + lead + content for GERMAN: {sum([len(ids) for ids in title_tokens_de['input_ids']]) + sum([len(ids) for ids in lead_tokens_de['input_ids']]) + total_content_tokens_de}")
        print(f"Total number of tokens in head + lead + content for FRENCH: {sum([len(ids) for ids in title_tokens_fr['input_ids']]) + sum([len(ids) for ids in lead_tokens_fr['input_ids']]) + total_content_tokens_fr}")

        # get average number of tokens in head + lead + content
        print(f"Average number of tokens in full text GERMAN: {(np.mean([len(ids) for ids in title_tokens_de['input_ids']]) * len(title_tokens_de['input_ids']) + np.mean([len(ids) for ids in lead_tokens_de['input_ids']]) * len(lead_tokens_de['input_ids']) + total_content_tokens_de)/len(dataset)}")
        print(f"Average number of tokens in full text FRENCH: {(np.mean([len(ids) for ids in title_tokens_fr['input_ids']]) * len(title_tokens_fr['input_ids']) + np.mean([len(ids) for ids in lead_tokens_fr['input_ids']]) * len(lead_tokens_fr['input_ids']) + total_content_tokens_fr)/len(dataset)}")


        #print total number of sentences
        print(f"Total number of sentences in content GERMAN: {sum_sentences_de}")
        print(f"Total number of sentences in content FRENCH: {sum_sentences_fr}")



    elif args.viz:
        # plot the score distribution
        n, bins, patches = plt.hist(dataset['score'], bins=100, density=False, color='#ADD8E6', alpha=0.7)
        
        # add mean line
        plt.axvline(x=78.64999999, color='grey', linestyle='--', linewidth=0.8, label=f'Top 15k cut: 78.64')
        plt.legend()
        
        plt.title('Document Similarity Score Distribution')
        plt.xlabel('Cosine similarity')
        plt.ylabel('Frequency')
        plt.show()

    elif args.extremes:
        # print the extremes
        print(dataset.sort_values(by='score', ascending=False).head(10))
        print(dataset.sort_values(by='score', ascending=True).head(10))

        # get mean of all scores
        mean_score = dataset['score'].mean()
        print(mean_score)
        # print 10 samples that are around the mean
        print(dataset[(dataset['score'] > mean_score - 0.00001*mean_score) & (dataset['score'] < mean_score + 0.00001*mean_score)].head(10))

    elif args.correlation_test:
        # subset of data with score > 46
        #dataset_high_score = dataset[dataset['score'] > 46]
        dataset_high_score = dataset[dataset['kendall_tau'].notna()]
        # drop rows with NaN in reorder_score
        #dataset_high_score = dataset_high_score[dataset_high_score['kendall_tau'].notna()]

        # get the pearson correlation between score and char_count_diff
        corr = dataset_high_score['score'].corr(dataset_high_score['kendall_tau'], method='pearson')
        print(corr)
        # get significance of the correlation
        corr, p_value = stats.pearsonr(dataset_high_score['score'], dataset_high_score['kendall_tau'])
        print(f"Correlation: {corr}, p-value: {p_value:.3f}")

    elif args.num_aligned_sentences:
        # print the number of aligned sentences
        aligned_sentences = pd.read_csv('data/sentence_alignments/mpnet_highest_match_non_sents_removed_sentence_length.tsv', sep="\t")
        print("aligned_sentences length:", len(aligned_sentences))
        # drop rows that have a score below 46
        aligned_sentences = aligned_sentences[aligned_sentences['score'] > 46]
        print("new aligned_sentences length:", len(aligned_sentences))
        # get the number of aligned sentences for each article
        num_aligned_sentences_de = aligned_sentences.groupby('aligned_article_id_de')['aligned_article_id_fr'].count()
        num_aligned_sentences_fr = aligned_sentences.groupby('aligned_article_id_fr')['aligned_article_id_de'].count()
        # get total number of sentences per article
        total_sentences_de = pd.read_csv('data/split_sentences/mpnet/de/split_sentences.tsv', sep="\t")
        total_sentences_fr = pd.read_csv('data/split_sentences/mpnet/fr/split_sentences.tsv', sep="\t")

        total_sentences_de['sentence_id'] = total_sentences_de['sentence_id'].astype(str).str.split('.').str[0]
        total_sentences_fr['sentence_id'] = total_sentences_fr['sentence_id'].astype(str).str.split('.').str[0]

        num_sentences_de = total_sentences_de.groupby('sentence_id')['sentence_id'].count()
        num_sentences_fr = total_sentences_fr.groupby('sentence_id')['sentence_id'].count()
        print(num_sentences_de)

        # find the corresponding articles and get the ratio
        # Initialize aligned_sents_ratio column with zeros
        dataset['aligned_sents_ratio_de'] = 0.0
        dataset['aligned_sents_ratio_fr'] = 0.0
        # Debug prints
        print("Number of articles in dataset:", len(dataset))
        print("Number of articles in num_aligned_sentences_de:", len(num_aligned_sentences_de))
        print("Number of articles in num_aligned_sentences_fr:", len(num_aligned_sentences_fr))
        print("Sample of dataset ids:", dataset['id_de'].head())
        print("Sample of num_aligned_sentences_de ids:", num_aligned_sentences_de.index[:5])
        
        # Check if ids match between dataframes
        for article_de, article_fr in zip(num_aligned_sentences_de.index, num_aligned_sentences_fr.index):
            # Convert ids to string for comparison
            article_de_str = str(article_de)
            article_fr_str = str(article_fr)
            
            if article_de_str in num_sentences_fr.index and article_fr_str in num_sentences_de.index:
                de_ratio = num_aligned_sentences_de[article_de] / num_sentences_fr[article_de_str]
                fr_ratio = num_aligned_sentences_fr[article_fr] / num_sentences_de[article_fr_str]
                
                print(f'article_de: {article_de}, ratio: {de_ratio}')
                print(f'article_fr: {article_fr}, ratio: {fr_ratio}')
                
                # Update values in dataset
                dataset.loc[dataset['id_fr'] == article_de, 'aligned_sents_ratio_de'] = de_ratio
                dataset.loc[dataset['id_de'] == article_fr, 'aligned_sents_ratio_fr'] = fr_ratio

            else:
                print(f"Missing article: de={article_de}, fr={article_fr}")
        
        print("\nFinal statistics:")
        print(dataset.head())
        print("Number of NaN values DE:", dataset['aligned_sents_ratio_de'].isna().sum())
        print("Mean ratio DE:", dataset['aligned_sents_ratio_de'].mean())
        print("Number of NaN values FR:", dataset['aligned_sents_ratio_fr'].isna().sum())
        print("Mean ratio FR:", dataset['aligned_sents_ratio_fr'].mean())

        # save the dataset
        dataset.to_csv('data/alignments/top15k/mpnet_align_sents_ratio.tsv', sep="\t", index=False)
    
    elif args.mono:
        aligned_sentences = pd.read_csv('data/sentence_alignments/mpnet_highest_match_non_sents_removed_sentence_length.tsv', sep="\t")
        # remove rows with score < 46
        aligned_sentences = aligned_sentences[aligned_sentences['score'] > 46]
        
        # split the sentence ids from id_de and id_fr
        aligned_sentences['id_de'] = aligned_sentences['id_de'].astype(str).str.split('.').str[1].astype(int)
        aligned_sentences['id_fr'] = aligned_sentences['id_fr'].astype(str).str.split('.').str[1].astype(int)
        
        # get all unique aligned_article_de and aligned_article_fr
        unique_aligned_article_de = aligned_sentences['aligned_article_id_de'].unique()
        unique_aligned_article_fr = aligned_sentences['aligned_article_id_fr'].unique()
        

        # Group by aligned_article_id_de and get sorted ids for each article
        grouped = aligned_sentences.groupby('aligned_article_id_de').agg({
            'id_de': list,
            'id_fr': list,
            'aligned_article_id_fr': 'first'  # Get corresponding fr article id
        })

        # Calculate kendall tau for each article pair
        kendall_taus = [stats.kendalltau(de_ids, fr_ids).statistic for de_ids, fr_ids in zip(grouped['id_de'], grouped['id_fr'])]

        df = pd.DataFrame({
            'article_id_de': grouped.index,
            'article_id_fr': grouped['aligned_article_id_fr'],
            'kendall_tau': kendall_taus
        })
        print(df.head())

        # rename column 'article_id_de' to 'id_fr'
        df = df.rename(columns={'article_id_de': 'id_fr'})
        # rename column 'article_id_fr' to 'id_de'
        df = df.rename(columns={'article_id_fr': 'id_de'})  

        merged = pd.merge(dataset, df, on=['id_de', 'id_fr'], how='left')
        print(merged.head())
        
        # save
        merged.to_csv('data/alignments/top15k/mpnet_align_sents_kendall_tau.tsv', sep="\t", index=False)


    elif args.length_correlation:

        aligned_sentences = pd.read_csv('data/sentence_alignments/mpnet_highest_match_non_sents_removed_sentence_length.tsv', sep="\t")
        # remove rows with score < 46
        aligned_sentences = aligned_sentences[aligned_sentences['score'] > 46]
        
        # Group by aligned_article_id_de and get sorted char_counts for each article
        grouped = aligned_sentences.groupby('aligned_article_id_de').agg({
            'char_count_de': list,
            'char_count_fr': list,
            'aligned_article_id_fr': 'first'  # Get corresponding fr article id
        })
        print(grouped.head())
        # Calculate pearson r for each article pair
        pearson_rs = [stats.pearsonr(de_counts, fr_counts).statistic  if len(de_counts) > 1 else 0 for de_counts, fr_counts in zip(grouped['char_count_de'], grouped['char_count_fr'])]

        df = pd.DataFrame({
            'article_id_de': grouped.index,
            'article_id_fr': grouped['aligned_article_id_fr'],
            'pearson_r': pearson_rs
        })  

        # rename column 'article_id_de' to 'id_fr'
        df = df.rename(columns={'article_id_de': 'id_fr'})
        # rename column 'article_id_fr' to 'id_de'
        df = df.rename(columns={'article_id_fr': 'id_de'})  

        # merge with dataset
        merged = pd.merge(dataset, df, on=['id_de', 'id_fr'], how='left')
        print(merged.head())

        # save
        merged.to_csv('data/alignments/top15k/mpnet_align_sents_pearson_r.tsv', sep="\t", index=False)
        
    elif args.vis_align:
        # plot the aligned_sents_ratio_de in correlation with the score as a scatter plot
        # Plot both DE and FR align ratios in one figure
        # drop rows with NaN in kendall_tau
        dataset = dataset[dataset['kendall_tau'].notna()]
        plt.scatter(dataset['score'], dataset['kendall_tau'], s=20, alpha=0.5, color='#ADD8E6')
        # plt.scatter(dataset['score'], dataset['kendall_tau_fr'], s=20, alpha=0.5, color='#FFB6C1', label='French')
        
        # Add smoothed trend lines
        scores_sorted = sorted(dataset['score'])
        
        # DE trend line
        z_de = np.polyfit(dataset['score'], dataset['kendall_tau'], 1)
        p_de = np.poly1d(z_de)
        plt.plot(scores_sorted, p_de(scores_sorted), "--", color='blue', alpha=0.8, label='Trend line', linewidth=1)

        
        # # FR trend line  
        # z_fr = np.polyfit(dataset['score'], dataset['kendall_tau_fr'], 1)
        # p_fr = np.poly1d(z_fr)
        # plt.plot(scores_sorted, p_fr(scores_sorted), "--", color='red', alpha=0.8, label='French trend', linewidth=1)

        plt.xlabel('Cosine Similarity')
        plt.ylabel('Kendall Tau')
        plt.title('Document Similarity Score vs. Monotonicity')
        
        # Add legend
        plt.legend(framealpha=1.0, loc='lower right')
        
        # Set x-axis limits to start at minimum score value
        plt.xlim(left=min(dataset['score']))
        
        plt.show()


if __name__ == "__main__":
    main()