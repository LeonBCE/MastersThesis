# USC X-24 Election Discourse Pipeline

This repository contains the code and small derived data files used for a thesis pipeline that transforms public Twitter/X discourse about the 2024 United States presidential election into state-level topic-salience indicators for PLS-SEM modelling.

The pipeline is split into five Colab notebooks and three R/Quarto files:

1. `notebooks/01_get_data.ipynb` cleans the raw Hugging Face parquet files, extracts usable United States state information, and saves the cleaned tweet corpus.
2. `notebooks/02_topic_modelling.ipynb` trains BERTopic on a large sample, assigns topics to the cleaned corpus, and exports the topic codebook.
3. `notebooks/03_sentiment_analysis.ipynb` scores sentiment and creates a diagnostic stance sample.
4. `notebooks/04_aggregation.ipynb` maps BERTopic topic IDs into macro and micro salience variables and exports the SEM datasets.
5. `notebooks/05_visualizations.ipynb` creates selected topic and state-level visualization figures.
6. `r_pls_sem/pls_sem_macro.qmd` estimates macro-topic PLS-SEM models.
7. `r_pls_sem/pls_sem_micro.qmd` estimates micro-topic and refined policy/conflict PLS-SEM models.
8. `r_pls_sem/pls_sem_bruteforce_exploratory.qmd` screens exploratory alternative model specifications.

## Repository Structure

```text
notebooks/                  Colab notebooks for the full Python pipeline
code_snippets/              Plain-text versions of the notebook cells
r_pls_sem/                  Quarto/R files for PLS-SEM modelling
data/                       Small final CSV exports used by the R models
outputs/                    Selected model output summaries and figures
figures/                    Selected visualization outputs
docs/                       Data availability and reproducibility notes
```

## Data

The raw tweet dataset is not included in this repository. It comes from the public Hugging Face dataset `deadbirds/usc-x-24-us-election-parquet`, which is shared under CC BY-NC-SA 4.0. Because the raw and intermediate tweet-level files are large and contain public social-media text, this repository includes only small derived files needed for transparency and SEM replication:

- `data/sem_dataset_macro.csv`
- `data/sem_dataset_micro.csv`
- `data/topic_codebook.csv`
- `data/topic_table_sorted_for_excel.txt`
- `data/macro_topic_mapping_for_excel.txt`

See `docs/DATA_AVAILABILITY.md` for the files that must be downloaded or regenerated if the full Colab pipeline is rerun from scratch.

## Running the Pipeline

The Python pipeline was designed for Google Colab with Google Drive mounted at:

```text
/content/drive/MyDrive/Thesis_Clean_Pipeline
```

The notebooks use Colab Secrets for the Hugging Face token:

```python
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"))
```

Do not hard-code Hugging Face tokens into notebooks.

The R/Quarto files can be run locally in RStudio after placing the final CSV datasets in the working directory expected by the scripts, or after updating the file paths at the top of the QMD files.

## Main Outputs

The final SEM input files are:

- `sem_dataset_macro.csv`: state-level macro-topic salience variables and Republican vote share.
- `sem_dataset_micro.csv`: state-level micro-topic salience variables and Republican vote share.

The most important final PLS-SEM outputs are saved under:

- `outputs/macro_model_outputs/`
- `outputs/micro_model_outputs/`
- `outputs/bruteforce_model_outputs/`

The final preferred compact model is the refined policy/conflict model in:

```text
outputs/micro_model_outputs/M10_refined_direct.txt
outputs/micro_model_outputs/M10_refined_direct.png
```

## Notes

Stance detection was retained as a diagnostic branch only. Validation showed that the zero-shot stance classifier did not recover political direction reliably enough, so stance is not used as a final SEM variable.

Sentiment was scored and inspected, but the final SEM models use topic salience as the central modelling input.

