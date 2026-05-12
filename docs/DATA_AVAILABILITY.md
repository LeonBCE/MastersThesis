# Data Availability

## Raw Dataset

The raw tweet data used in the thesis comes from:

```text
https://huggingface.co/datasets/deadbirds/usc-x-24-us-election-parquet
```

The dataset page states that it is adapted and shared under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License (CC BY-NC-SA 4.0).

The raw Hugging Face parquet files are not included in this repository.

## Included Data Files

The repository includes small derived outputs that are sufficient for inspecting the final SEM inputs and rerunning the R/Quarto PLS-SEM models:

```text
data/sem_dataset_macro.csv
data/sem_dataset_micro.csv
data/topic_codebook.csv
data/topic_table_sorted_for_excel.txt
data/macro_topic_mapping_for_excel.txt
```

These files are state-level or topic-level summaries rather than raw tweet-level text.

## Files Needed From Google Drive If Rerunning the Full Pipeline

If rerunning the full pipeline from notebook 01, no intermediate files need to be downloaded from Drive first. The notebooks regenerate the downstream files in order from the raw Hugging Face source.

If rerunning only later notebooks, the following intermediate files are needed in:

```text
/content/drive/MyDrive/Thesis_Clean_Pipeline/data/interim/
```

Required for notebook 02 if notebook 01 is skipped:

```text
english_state_cleaned.parquet
```

Required for notebook 03 if notebook 02 is skipped:

```text
tweets_with_topics.parquet
```

Required for notebook 04 and sentiment/salience visualizations if notebook 03 is skipped:

```text
tweets_with_topics_scores.parquet
```

Optional, only if reusing the exact BERTopic training sample:

```text
topic_training_sample.parquet
```

These intermediate parquet files are intentionally not included in the GitHub folder because they are large and contain tweet-level text.

## External State-Level Data

Notebook 04 fetches state-level election results and contextual variables from public web/API sources during aggregation:

- 2024 presidential election state results from a public results table.
- U.S. Census Bureau 2022 ACS 5-year API variables for state-level demographic controls.
- A public state-level urbanization table.

The final SEM exports in this repository keep only state name, Republican vote share, and topic salience variables.

