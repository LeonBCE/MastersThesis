# Reproducibility Notes

## Random Seeds

The topic-modelling sample and diagnostic samples use fixed random seeds where possible. BERTopic still includes stochastic steps through dimensionality reduction and clustering, so exact topic IDs should always be checked against the exported `topic_codebook.csv` before rerunning aggregation.

## GPU Notes

The Colab notebooks were written to use an A100 GPU when available. The most GPU-intensive steps are sentence embedding, sentiment scoring, and stance scoring. UMAP and HDBSCAN in the BERTopic stage are CPU-bound in the current implementation, so GPU memory usage can drop during dimensionality reduction and clustering.

## Topic IDs

Topic IDs are not stable across independent BERTopic training runs. The final mapping in notebook 04 is based on the included `data/topic_codebook.csv`. If BERTopic is retrained, inspect the new topic codebook and update the macro/micro mappings before creating SEM datasets.

## Stance

Stance detection was attempted with a zero-shot NLI model, but validation showed that the output was not reliable enough for final modelling. The stance output is kept only as diagnostic evidence and should not be used as a final SEM predictor without additional supervised validation.

## PLS-SEM

The R/Quarto models use the final state-level SEM CSVs. Bootstrapping results should be interpreted from the tabular model outputs rather than only from stars shown in plotted diagrams.

