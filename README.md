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

Column filtering, text cleaning, stop-word removal, lemmatization, tokenization, converting to vectors and creating word embeddings

### Model

Bi-directional LSTM w/ Attention

### Evaluation

Accuracy metric + confusion matrix. Compared with fine-tuned BERT model (Refer [Model comparison section](#model-comparison) for details)

## How to run

1. Install dependencies

```bash
pip3 install -r requirements.txt
```

2. Run pyspark pre-processing job to generate datasets + embeddings for training and testing (stored in `data/`)

```bash
python3 preprocess.py
```

3. Train model (optional, as trained model is already present in `models/`. Skip to next step)

```bash
python3 train.py
```

4. Evaluate model

```bash
python3 test.py
```

## Model comparison

We have evaluated this model against the state-of-the-art Transformer-based model, BERT, trained via transfer-learning and fine-tuned on our dataset. \
This approach doesn't require any pre-processing on the text, since BERT does it's own tokenization and embedding. \
The complete code for our Bi-directional LSTM model and the BERT model as well as the evaluation of those models can be found in the notebook `hw3.ipynb` (or access it here: <https://github.com/JediRhymeTrix/COSC-6339-HW3/blob/master/hw3.ipynb>)
