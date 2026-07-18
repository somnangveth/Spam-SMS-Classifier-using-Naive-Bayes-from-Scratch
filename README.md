# Spam SMS Classifier using Naive Bayes (From Scratch)

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Stopwords-brightgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-MultinomialNB-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A hand-built implementation of a **Naive Bayes spam classifier**, developed from first principles before comparing against scikit-learn's built-in `MultinomialNB`. This project walks through the full pipeline — tokenization, vocabulary building, spamicity/hamicity calculation, Laplace smoothing, stop-word removal, and Bayesian classification — without relying on an ML library for the core algorithm.

---

## Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Example Output](#-example-output)
- [From Scratch vs. scikit-learn](#-from-scratch-vs-scikit-learn)
- [Resources](#-resources)
- [License](#-license)

---

## Overview

This project builds a **spam vs. ham (not spam)** SMS classifier using the **Naive Bayes algorithm**, implemented manually to understand the underlying math before using scikit-learn's optimized version.

Key concepts covered:
- Text preprocessing & tokenization
- Building word vocabularies from labelled training data
- Calculating **spamicity** and **hamicity** (word-level conditional probabilities)
- **Laplace (add-one) smoothing** to handle zero-frequency words
- Dropping unseen words vs. smoothing — and why
- Stop-word removal to reduce noise
- Applying **Bayes' Theorem** to classify unlabelled messages
- Benchmarking against scikit-learn's `MultinomialNB`

---

## Dataset

**[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)** (Kaggle / UCI)

A set of 5,574 SMS messages labelled as either:
- `spam` — unsolicited/promotional messages
- `ham` — legitimate, non-spam messages

| Column | Description |
|--------|-------------|
| `label` | `spam` or `ham` |
| `message` | Raw SMS text |

> Download `spam.csv` from the Kaggle link above and place it in the `dataset/` folder before running the scripts.

---

## How It Works

1. **Preprocessing** — lowercase all messages, split into word tokens
2. **Train/test split** — stratified split to preserve spam/ham ratio
3. **Vocabulary building** — extract unique words separately from spam and ham training messages
4. **Probability calculation**:
   - `P(word | spam)` → *spamicity*
   - `P(word | ham)` → *hamicity*
   - Applies **Laplace smoothing**: `(count + 1) / (total + 2)`
5. **Unseen word handling** — words not seen in training are dropped (not smoothed), since there's no basis to estimate their spam/ham likelihood
6. **Stop-word removal** — filters out uninformative high-frequency words (e.g., "the", "of", "your") using NLTK/sklearn stop-word lists
7. **Classification** — applies Bayes' Theorem word-by-word, then combines probabilities to output a final SPAM/HAM prediction with a confidence score

---

## Project Structure

```
.
├── dataset/
│   └── spam.csv                # Kaggle SMS Spam Collection dataset
├── naive_bayes_scratch.py      # From-scratch implementation
├── naive_bayes_sklearn.py      # scikit-learn MultinomialNB comparison
└── README.md
```

---

## Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn nltk
```

### NLTK Stopwords Setup

```python
import nltk
nltk.download('stopwords')
```

>  If you hit an SSL `CERTIFICATE_VERIFY_FAILED` error on macOS, run the *Install Certificates* script bundled with your Python installation, or use `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` instead — no download required.

---

## Usage

**Run the from-scratch classifier:**
```bash
python naive_bayes_scratch.py
```

**Run the scikit-learn comparison:**
```bash
python naive_bayes_sklearn.py
```

**Test a custom message:**
```python
new_test_email = "urgent free entry to win a prize call now to claim your cash reward"
Bayes(new_test_email)
```

---

## Example Output

```
All word probabilities for this sentence:
[0.981, 0.994, 0.950, 0.988, 0.985, 0.996, 0.990, 0.983, 0.900]

Email is SPAM: with spammy confidence of 78.64%
```

---

## From Scratch vs. scikit-learn

| | From Scratch | scikit-learn (`MultinomialNB`) |
|---|---|---|
| Vocabulary building | Manual dictionaries | `CountVectorizer` |
| Smoothing | Manual `+1` Laplace smoothing | `alpha` parameter (auto) |
| Unseen word handling | Manual filtering | Automatic via `.transform()` |
| Stop-word removal | Manual intersection with NLTK list | `stop_words='english'` param |
| Probability combination | Manual multiplication (risk of underflow) | Computed safely in log-space |
| Speed | Slow, educational | Fast, production-ready |

Building it from scratch first makes the sklearn version far easier to trust and reason about — every parameter maps to something you've already implemented by hand.

---

## Resources

- Dataset from Kaggle: [Dataset — SMS Spam Collection (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/data)
- Naive Bayes Resource: [Naive Bayes Spam Filter From Scratch — Towards Data Science](https://towardsdatascience.com/naive-bayes-spam-filter-from-scratch-12970ad3dae7/)
- ScikitLearn Docs: [scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

## License

This project is licensed under the [MIT License](LICENSE).
