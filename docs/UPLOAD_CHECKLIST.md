# GitHub Upload Checklist

Before uploading this folder to GitHub:

1. Check that the repository name is spelled the way you want it. The local folder is currently named `ghithub` because that was the requested folder name.
2. Do not add raw Hugging Face parquet files.
3. Do not add intermediate tweet-level parquet files from Google Drive.
4. Do not add thesis Word drafts.
5. Do not add `.RData`, `.Rhistory`, or Colab checkpoints.
6. Keep Hugging Face access through Colab Secrets only.
7. If GitHub Desktop or the command line asks about large files, do not commit them unless they are small final CSVs or selected output figures already included here.

## Included Intentionally

The repository includes:

- Colab notebooks, with outputs stripped.
- Plain-text notebook snippets.
- R/Quarto PLS-SEM files.
- Final macro and micro SEM datasets.
- Topic codebook and topic mapping helper files.
- Selected model output summaries.
- Selected visualization PNGs.
- Diagnostic stance-validation files documenting why stance was not used in the final SEM models.

## Not Included Intentionally

The repository does not include:

- Raw tweet parquet files.
- Full tweet-level cleaned/scored parquet files.
- Hugging Face tokens.
- Thesis Word drafts.
- Large ZIP archives.
- RStudio session files.
