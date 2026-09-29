# MobileRec: Neural Networks for Mobile App Recommendation

An end-to-end recommender systems study comparing **four neural architectures** with **two non-neural baselines** on **19.3 million Google Play reviews**.

Developed by **Milena Đerić** for the bachelor's thesis *Application of Neural Networks in Recommender Systems*, Faculty of Organizational Sciences, University of Belgrade.

**Research question:** Can neural models improve next-app ranking over popularity and item similarity when all models learn from the same implicit interactions?

The project covers data preparation, sparse interaction modeling, PyTorch training, validation-based tuning, and evaluation beyond aggregate ranking scores. It also documents a practical outcome: greater model complexity does not necessarily produce better recommendations.

[Results](#results) · [Evaluation protocol](#evaluation-protocol) · [Notebooks](#notebook-guide) · [Run locally](#run-locally)

## Data and task

The [MobileRec dataset](https://huggingface.co/datasets/recmeapp/mobilerec) contains app reviews and a separate app-metadata table. Each review is treated as an observed user–app interaction, regardless of its rating.

The task is to rank the **next app a user will review**. An observed review is not assumed to mean that the user liked, installed, or continued using the app.

| Dataset property | Value |
| --- | ---: |
| Recorded interactions | 19,297,019 |
| Users | 700,111 |
| Apps in the source dataset | 10,173 |
| Apps observed in training | 10,172 |
| Training interactions | 17,896,797 |
| Unique training user–app pairs | 17,748,311 |
| Interaction-matrix density | 0.2492% |

Ratings and review text are not used as model inputs. App metadata support exploratory and category-level analysis rather than recommendation scoring.

## Evaluation protocol

![Chronological split within each user history, with training-only input for validation and test scoring.](docs/images/evaluation_protocol.png)

For each user, interactions are ordered by date, with original row order breaking same-day ties. All but the last two interactions form the training history; the second-to-last is the validation target, and the last is the test target.

- User mappings, app mappings, popularity counts, and the candidate set are built from training data only.
- Hyperparameters and checkpoints are selected by **validation NDCG@10**. Test targets are reserved for final evaluation.
- Both validation and test scoring use **training history only**. The validation interaction is not added to the test-time input.
- Models rank the full set of **10,172 training apps**, without sampled evaluation negatives. Previously seen apps remain eligible because repeat interactions can be targets.
- Three validation targets and 15 test targets refer to an app absent from training. Ranking metrics therefore use **700,108 validation targets** and **700,096 test targets**.

Each user has one relevant target, so Hit@K and Recall@K are equal. NDCG@K also rewards placing that target nearer the top. Extended evaluation considers K = 1, 5, 10, 15, and 20, user history length, target popularity, app category, and catalog coverage.

This is a **per-user chronological split**, not a single global time cutoff.

## Models and implementation

| Model | Approach |
| --- | --- |
| Popularity | Same list of the most frequent training apps for every user |
| ItemKNN | Item-based collaborative filtering with cosine similarity |
| AutoRec | User-history autoencoder adapted to binary interactions with weighted binary cross-entropy |
| MultiVAE | Variational autoencoder with reconstruction loss and annealed KL regularization |
| NeuMF | GMF and MLP branches, adapted to pairwise BPR ranking |
| LightGCN | Embedding propagation over the user–app graph with BPR training |

**Engineering work demonstrated**

- Batch-based validation and cleaning of more than 19 million interactions; compressed Parquet output and DuckDB queries.
- A shared SciPy CSR interaction matrix, avoiding a dense 700,111 × 10,172 allocation.
- PyTorch training loops, negative sampling, checkpoint selection, early stopping, and hyperparameter experiments.
- LightGCN memory and runtime optimization through a sparse CSR graph and fewer, larger training batches.
- Consistent evaluation of saved checkpoints, with analysis of popularity concentration and catalog coverage.

The experiments were developed on an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM. A fixed seed of 42 is used; this does not guarantee identical results across hardware and library versions.

## Results

<!-- RESULTS:START -->
> **Results status: pending rerun verification.** These numbers describe the saved evaluation snapshot; the refreshed run has not yet been confirmed.

| Model | Test Hit@10 | Test NDCG@10 | Catalog coverage@10 |
| --- | ---: | ---: | ---: |
| ItemKNN | 1.6872% | 0.008256 | 10.25% |
| Popularity | 1.5051% | 0.007762 | 0.10% |
| NeuMF | 1.5044% | 0.007754 | 0.12% |
| LightGCN | 1.4648% | 0.007447 | 43.58% |
| MultiVAE | 1.3860% | 0.006912 | 0.12% |
| AutoRec | 1.0557% | 0.004554 | 1.18% |

NDCG values are shown on the 0–1 scale; Hit@10 and coverage are percentages.

- **Best ranking:** ItemKNN leads this snapshot. NeuMF has the highest NDCG@10 among the neural models.
- **Broadest catalog:** LightGCN recommends 4,433 distinct apps across users' Top-10 lists.
- **Popularity concentration:** 99.79% of NeuMF recommendation positions belong to the ten most popular training apps.
<!-- RESULTS:END -->

![Side-by-side comparison of test NDCG at 10 and catalog coverage for all six models.](docs/images/model_comparison.png)

Catalog coverage is the fraction of candidate apps appearing in at least one user's Top-10 list. It measures breadth across users, not whether those recommendations are relevant.

The table and chart come from the same [saved results snapshot](docs/results/overview.csv). Additional coverage statistics are available in [catalog.csv](docs/results/catalog.csv), with source hashes and status in [provenance.json](docs/results/provenance.json).

## Notebook guide

Run the main notebooks in this order. Each stage saves artifacts needed by the next.

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | [Data analysis and preparation](700k%20test/01_mobilerec_data_analysis_and_preparation.ipynb) | Inspect interactions and metadata; save prepared Parquet files |
| 2 | [Chronological split](700k%20test/02_mobilerec_chronological_data_split.ipynb) | Build train/validation/test sets, mappings, and popularity counts |
| 3 | [Baselines](700k%20test/03_mobilerec_popularity_and_itemknn_baselines.ipynb) | Create the sparse matrix; evaluate Popularity and tune ItemKNN |
| 4 | [Autoencoders](700k%20test/04_mobilerec_autorec_and_multivae.ipynb) | Train and tune AutoRec and MultiVAE |
| 5 | [NeuMF](700k%20test/05_mobilerec_neumf.ipynb) | Train and tune the GMF–MLP model |
| 6 | [LightGCN](700k%20test/06_mobilerec_lightgcn.ipynb) | Train and tune the optimized graph model |
| 7 | [Extended evaluation](700k%20test/07_mobilerec_extended_evaluation.ipynb) | Reload checkpoints and compare ranking, user groups, and coverage |

Two additional notebooks preserve development versions: `00_mobilerec_lightgcn_pre_optimization.ipynb` and `05a_mobilerec_neumf_development.ipynb`. They are **not extra pipeline steps**. They use some of the same output locations as the main notebooks; do not run them as part of the main sequence.

Markdown explanations, printed messages, and figure labels are in English. Some code comments remain in Serbian.

## Run locally

### Environment

The saved notebooks report Python 3.13.9. The following is a setup starting point, not a fully locked environment specification.

From Anaconda Prompt:

```shell
conda create -n mobilerec python=3.13
conda activate mobilerec
python -m pip install notebook ipykernel numpy pandas scipy scikit-learn matplotlib duckdb datasets pyarrow
python -m ipykernel install --user --name mobilerec --display-name "Python (mobilerec)"
```

Install PyTorch using the command for your operating system and compute platform from the [official installation selector](https://pytorch.org/get-started/locally/). The full neural experiments were run with CUDA; although the notebooks select CPU when CUDA is unavailable, full-scale CPU training has not been benchmarked here.

From the project directory containing this README:

```shell
cd "700k test"
jupyter notebook
```

Select the **Python (mobilerec)** kernel and run the main notebooks in order. [Jupyter's installation guide](https://jupyter.org/install) also documents the Notebook launcher.

### Data and generated files

Notebook 1 downloads the two dataset components separately:

```python
from datasets import load_dataset

interactions = load_dataset("recmeapp/mobilerec", data_dir="interactions")
metadata = load_dataset("recmeapp/mobilerec", data_dir="app_meta")
```

Allow several GB of disk space for the download cache, prepared data, temporary files, sparse matrices, and checkpoints. Paths in the notebooks are relative to `700k test`, with generated files written under `artifacts/`.

Existing preparation and split files may be reused rather than rebuilt. Notebook 7 also caches per-user evaluation results: set **`force_recompute = True`** after retraining or changing the evaluation inputs, then run the complete evaluation notebook. Check its comparison against the saved model results before refreshing this README.

### Refresh the README figures

From the directory containing this README:

```shell
python scripts/build_readme_assets.py --refresh-source
```

This copies the small evaluation CSVs into `docs/results/`, records their hashes, and rebuilds the results table and both figures without loading or training models. The default output remains labeled as pending rerun verification. After the rerun and consistency checks have been completed, use:

```shell
python scripts/build_readme_assets.py --refresh-source --confirmed
```

To regenerate figures from the existing documentation snapshot, run the script without arguments.

## Limitations and next experiments

Each neural configuration was trained with one seed, so results do not estimate variability across independent runs. Hyperparameter searches and compute budgets differ between model families; this is not an exhaustive comparison of their best possible performance.

MobileRec records reviews rather than impressions, installations, usage sessions, or exposure. Unobserved apps are not verified dislikes. Timestamps have day-level resolution, and a per-user split does not enforce global temporal isolation.

Useful next experiments include repeated seeds, alternative negative sampling and ranking losses, attention-based item similarity, and approaches that reduce popularity concentration. Historical metadata or richer interaction logs would support additional content-based and sequential models.

## References

- [MobileRec dataset and paper citation](https://huggingface.co/datasets/recmeapp/mobilerec) — Maqbool et al., 2023.
- [AutoRec](https://doi.org/10.1145/2740908.2742726) — Sedhain et al., 2015.
- [Variational Autoencoders for Collaborative Filtering](https://doi.org/10.1145/3178876.3186150) — Liang et al., 2018.
- [Neural Collaborative Filtering](https://doi.org/10.1145/3038912.3052569) — He et al., 2017.
- [LightGCN](https://doi.org/10.1145/3397271.3401063) — He et al., 2020.
- [Bayesian Personalized Ranking](https://arxiv.org/abs/1205.2618) — Rendle et al., 2009.
