# COSC-6339-HW3 - Team 12

## Team Members

Kapoor, Kartik  2462 \
Nham, Bryan 2494 \
Panyala, Sukrutha 8740 \
Vohra, Vedant 2889

## Dataset

`amazon_reviews_grocery.tsv`

## Approach

### Preprocessing (using PySpark)

Column filtering, mapping categorical values to numerical, text cleaning, stop-word removal, lemmatization, tokenization, converting to vectors and creating word embeddings

### Models

**Bi-directional LSTM w/ Attention**

1. A model that only uses the review text as input
2. A model that uses the review text as well as some of the other numerical/categorical features as input (`helpful_votes, total_votes, vine, verified_purchase`) \
    -> The additional features are added as a second Input layer which is concatenated with the output from the LSTM, just before the Dense layers.

### Evaluation

Accuracy metric + confusion matrix. Compared with fine-tuned BERT model (Refer [Model comparison section](#model-comparison) for details)

## How to run

1. Install dependencies

```bash
pip3 install -r requirements.txt
```

2. Run pyspark pre-processing job to generate datasets + embeddings for training and testing (stored in `data/`)

```bash
spark-submit preprocess.py <absolute_file_path>
```

  (`absolute_file_path` path here can be a hdfs path)

3. Train models (optional, as trained models are already present in `models/`. Skip to next step)

```bash
python3 train.py
```

4. Evaluate models

```bash
python3 test.py
```

## Model comparison

We have evaluated this model against the state-of-the-art Transformer-based model, BERT, trained via transfer-learning and fine-tuned on our dataset. \
This approach doesn't require any pre-processing on the text, since BERT does it's own tokenization and embedding. \
The complete code for our Bi-directional LSTM model and the BERT model as well as the evaluation of those models can be found in the notebook `hw3.ipynb` (or access it here: <https://github.com/JediRhymeTrix/COSC-6339-HW3/blob/master/hw3.ipynb>)
