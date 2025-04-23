# _20min-XD_: A Comparable Corpus of Swiss News Articles

This repository contains the data and code for the scraping and postprocessing of the dataset as well as the experiments and analyses described in the paper "20min-XD: A Comparable Corpus of Swiss News Articles" (TODO: add link).

_20min-XD_ (**20 Min**uten **cross**-lingual **d**ocument-level) is a comparable corpus of Swiss news articles in German and French, collected from the online editions of 20 Minuten and 20 minutes between 2015 and 2024. The dataset consists of 15,000 semantically aligned German and French article pairs. Unlike parallel corpora, _20min-XD_ captures a broad spectrum of cross-lingual similarity, ranging from near-translations to related articles covering the same event. This dataset is intended for non-commercial research use only – please refer to the accompanying license/copyright notice for details.

In addition, the data is available on [Hugging Face](https://huggingface.co/datasets/ZurichNLP/20min-XD), both in a document and sentence aligned version.

## Usage

Requirements:

* Python == 3.11

```bash
pip install -r reqs.txt
```

### Scraping articles

```bash
python scripts/scrape_articles.py
```

### Postprocessing scraped articles

```bash
python scripts/clean_scraped_dataset.py --clean-articles
```

```bash
python scripts/clean_scraped_dataset.py --drop-dates
```

### Embed texts

```bash
python scripts/get_embeddings.py data/scraped_articles/fr/articles_cleaned_dates.tsv sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

```bash
python scripts/get_embeddings.py data/scraped_articles/de/articles_cleaned_dates.tsv sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

### Get similarity scores

```bash
python scripts/get_simscores.py data/embeddings/mpnet sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

### Align articles

```bash
python scripts/align_articles.py data/simscores/mpnet data/scraped_articles --align_type best
```

### Postprocess aligned articles and cut to top 15k

```bash
python scripts/get_top15k.py data/alignments/mpnet
```

At this point, the document-level data is ready for use under `data/alignments/[model_name]`.

### Split into sentences

```bash
python scripts/align_sentences.py data/alignments/top15k/mpnet.tsv --split_sentences
```

### Embed sentences

```bash
python scripts/align_sentences.py data/split_sentences/mpnet/fr/split_sentences.tsv --embed_sentences
```

### Get similarity scores for sentences

```bash
python scripts/align_sentences.py data/embedded_sentences/mpnet/fr/split_sentences.tsv --get_simscores
```

### Align sentences

```bash
python scripts/align_sentences.py data/sentence_simscores/mpnet.pkl --get_sentence_alignments
```

### Postprocess sentence alignments

```bash
python scripts/post_process_aligned_sentences.py data/sentence_alignments/mpnet.tsv --get_highest_match && \
python scripts/post_process_aligned_sentences.py data/alignments/mpnet_highest_match.tsv --remove_non_sents && \
python scripts/post_process_aligned_sentences.py data/sentence_alignments/mpnet_highest_match_non_sents_removed.tsv --get_sentence_length
```

At this point, the sentence-level data is ready for use under `data/alignments/[model_name]_non_sents_removed_sentence_length.tsv`.


## Reproduce experiments

Repeat the steps above with the desired set of models on the validation set under `data/val/val_set.tsv` until the similarity scores are computed.

Then, run the following script to get the best threshold for each mode with different alignment methods.

```bash
python scripts/get_threshold.py data/val/val_simscores.pkl --align_type best; 
python scripts/get_threshold.py data/val/val_simscores.pkl --align_type best_fr; 
python scripts/get_threshold.py data/val/val_simscores.pkl --align_type best_de;
python scripts/get_threshold.py data/val/val_simscores.pkl --align_type best_both;
python scripts/get_threshold.py data/val/val_simscores.pkl --align_type above_threshold
```


## Reproduce analyses

The following script provides a brief overview of the dataset statistics:
```bash
python scripts/get_dataset_statistics.py data/alignments/top15k/mpnet.tsv
```

Adding the `--viz` flag plots the similarity score distribution of the document-level dataset.

Adding the `--num_aligned_sentences` flag computes the AlignRatio for each article, saves it to `data/alignments/top15k/mpnet_align_sents_ratio.tsv` and prints the mean AlignRatio mean for each language.

Adding the `--mono` flag computes the Kendall Tau correlation (interpreted as "translation" monotonicity) between the sentence alignments of the DE and FR articles, saves it to `data/alignments/top15k/mpnet_align_sents_kendall_tau.tsv`.

Adding the `--length_correlation` flag computes the Spearman correlation between the sentence lengths of the sentences for each article pair and saves it to `data/alignments/top15k/mpnet_align_sents_pearson_r.tsv`.

Adding the `--vis_align` flag plots the Kendall Tau correlation and the Spearman correlation as a scatter plot. Adjust dataset_path to the file containing the correlations that you want to plot.


## Copyright notice

The resulting dataset is released with the following copyright notice:

### German / Deutsch (original):

> © 2025. TX Group AG / 20 Minuten.
> Dieser Datensatz enthält urheberrechtlich geschütztes Material von TX Group AG / 20 Minuten. Er wird ausschliesslich für nicht-kommerzielle wissenschaftliche Forschungszwecke bereitgestellt. Jegliche kommerzielle Nutzung, Vervielfältigung oder Verbreitung ohne ausdrückliche Genehmigung von TX Group AG / 20 Minuten ist untersagt.


### English / Englisch:

> © 2025. TX Group AG / 20 Minuten.
> This dataset contains copyrighted material from TX Group AG / 20 Minuten. It is provided exclusively for non-commercial scientific research purposes. Any commercial use, reproduction, or distribution without explicit permission from TX Group AG / 20 Minuten is prohibited.


## Citation

If you use the dataset, please cite the following paper:

```
TODO: add citation
```




