# Comment Category Prediction

An end-to-end machine learning pipeline for classifying online comments into categories, built for the Kaggle *Comment Category Prediction Challenge* (\~198K training comments, \~102K test comments).

### Overview

The task is a multi-class, heavily imbalanced text classification problem. Each comment comes with metadata (emoticon counts, identity-related categorical fields such as race/religion/gender, a disability flag, and two pre-engineered numeric signals `if_1`, `if_2`). The pipeline covers EDA, feature engineering, preprocessing, model training, and ensembling.

### Approach

- **EDA:** Analyzed class imbalance, sparsity of identity columns, comment length distributions per label, and feature correlations (`if_2` was the strongest numeric signal at \~0.23).
- **Feature selection with reasoning:** Dropped `upvote`, `downvote`, `created_date`, and `post_id`. Votes reflect community agreement rather than comment content (and risk leakage); time features showed near-zero correlation with the label.
- **Leakage prevention:** Stratified 80/20 train/validation split done on raw data *before* any feature engineering; the preprocessor is fit on training data only.
- **Feature engineering:** Comment length, word count, capital-letter ratio, punctuation count, URL flag, and total emoticon count.
- **Text cleaning:** Lowercasing, URL/HTML removal, whitespace normalization.
- **Preprocessing:** A single `ColumnTransformer` combining TF-IDF (50K features, unigrams + bigrams, sublinear TF), One-Hot Encoding for categorical columns, and StandardScaler for numeric features.
- **Imbalance handling:** `class_weight='balanced'` across all models.

### Models and Results (validation set)

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| LightGBM (early stopping) | 0.9000 | 0.7896 |
| Logistic Regression | 0.8996 | 0.7986 |
| Linear SVC | 0.7880 | 0.6878 |
| **Ensemble (LightGBM + LR, 50/50 probability average)** | **0.9064** | **0.8089** |

The final submission uses the soft-voting ensemble of LightGBM and Logistic Regression.

### Tech Stack

Python, pandas, NumPy, scikit-learn, LightGBM, SciPy, Matplotlib, Seaborn

### Repository Contents

- `notebook.ipynb`: full pipeline from EDA to submission generation
- `README.md`: project overview

### How to Run

1. Download the dataset from the Kaggle competition page and update the file paths in the notebook (currently set to `/kaggle/input/...`).
2. Install dependencies: `pip install pandas numpy scikit-learn lightgbm matplotlib seaborn scipy`
3. Run the notebook top to bottom to reproduce results and generate `submission.csv`.
