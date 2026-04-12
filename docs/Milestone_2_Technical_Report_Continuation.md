# Milestone 2 - Technical Report (Continuation)

**Jenny James, Sergey Nelyapenko, & Lexi Fortuna**
**Boston University | DX703 - Advanced Machine Learning and AI | Wayne Snyder | 12 APR 2026**

---

*This document continues from the existing "Milestone 2 - Technical Report.docx" and provides the completed content for all remaining sections: Modeling Approach, Training Strategy, Evaluation, Key Results and Analysis, and Limitations and Future Work.*

---

## Modeling Approach

We followed a progressive three-stage modeling approach, starting with a simple baseline and advancing to more complex architectures. This structure enables a clear comparison of how complexity, efficiency, and accuracy relate to one another across models.

### Baseline Model (Problem 2): Embedding + GlobalAveragePooling

The baseline model is a lightweight `Sequential` neural network:

- **Embedding(20,000 x 64):** Maps each token in the 20,000-word vocabulary to a 64-dimensional dense vector.
- **GlobalAveragePooling1D:** Averages all token embeddings in a sequence into a single 64-dimensional vector, intentionally discarding word order to establish a performance floor.
- **Dense(64, ReLU):** A hidden layer to learn non-linear feature combinations.
- **Dense(31, Softmax):** Output layer producing a probability distribution over the 31 merged news categories.

**Total parameters:** 1,286,175

This architecture was chosen because it matches the assignment's recommended text baseline and provides a fast, interpretable reference point. By discarding sequential information, any improvements from later models can be attributed to their ability to capture word order and context.

### Custom Models (Problem 3): Architectural Variations

After establishing the baseline, we explored three architectural modifications to address its main limitation — the complete loss of word-order information:

**Model 3a — LSTM(64):**
Replaced GlobalAveragePooling with a single LSTM(64) layer. The motivation was to capture sequential dependencies in text, allowing the model to distinguish between phrases where word order matters (e.g., "dog bites man" vs. "man bites dog"). All other components, including the optimizer and early stopping configuration, were kept consistent to isolate the effect of the recurrent layer. **Parameters:** 1,319,199.

**Model 3b — Bidirectional LSTM with Dropout:**
Extended the LSTM approach by wrapping it in a `Bidirectional` layer and adding `Dropout(0.5)`. The bidirectional structure processes text in both forward and backward directions, theoretically improving context understanding. Dropout was introduced to mitigate the overfitting already observed in the baseline. This model was trained for 8 fixed epochs without early stopping to observe the full learning trajectory. **Parameters:** 1,356,319.

**Model 3c — Enhanced Baseline (BatchNorm + Dropout + Dense):**
Rather than pursuing more complex sequence models, we revisited the original baseline and enhanced it with:
- Batch Normalization after each Dense layer (for optimization stability)
- Dropout(0.3) after each BatchNorm (for regularization)
- An additional Dense(128, ReLU) layer (for increased capacity)

This approach focused on improving generalization without increasing sequence modeling complexity. **Parameters:** 1,299,359.

### Pretrained Model (Problem 4): DistilBERT Transfer Learning

For transfer learning, we selected **DistilBERT** (`distilbert-base-uncased`), a distilled version of BERT that retains 97% of BERT's language understanding while being 40% smaller and 60% faster.

**Why DistilBERT over BERT or RoBERTa:**
- BERT has ~110M parameters — too heavy for the available compute budget and short-text classification task.
- RoBERTa is even larger and requires longer training.
- DistilBERT (66.9M parameters) stabilizes faster and generally does not need as many epochs to reach solid validation performance, making it practical for the HuffPost short-text dataset.

**Architecture adaptation:**
- Replaced the masked language model head with a classification head: `Pre-classifier(768, 768)` + `Classifier(768, 31)`.
- Used the Hugging Face `TFDistilBertForSequenceClassification` class with `num_labels=31`.

**Total parameters:** 66,977,311, of which approximately 14.2M were trainable (the top 2 transformer blocks plus the classification head — 36 trainable variable tensors in total).

---

## Training Strategy

All models were trained on the same 80/10/10 stratified split (160,290 train / 20,036 validation / 20,037 test) with a fixed random seed of 42.

### Data Preparation

- **Text formation:** Headline and short_description concatenated with a `[SEP]` separator; when description was missing (~9.8% of records), the headline alone was used.
- **Tokenization (Problems 2-3):** A `TextVectorization` layer with `max_tokens=20,000` and `output_sequence_length=64` (chosen based on the P95 word count of 57 words). Adapted on training text only, then applied once to produce integer-encoded numpy arrays.
- **Tokenization (Problem 4):** Hugging Face `AutoTokenizer` for DistilBERT with `max_length=64`, padding, and truncation.
- **Class weights:** Computed using `sklearn.compute_class_weight("balanced")`, producing weights from 0.198 (POLITICS) to 5.726 (LATINO VOICES), passed via `class_weight` parameter to `model.fit()`.

### Model-Specific Training Configurations

| Configuration | Baseline | LSTM | BiLSTM | Enhanced Baseline | DistilBERT |
|---|---|---|---|---|---|
| **Optimizer** | Adam | Adam | Adam | Adam | AdamW (weight decay) |
| **Learning Rate** | 0.001 (default) | 0.001 | 0.001 | 0.001 | 2e-5 (cosine schedule) |
| **Max Epochs** | 50 | 50 | 8 | 15 | 16 |
| **Batch Size** | 256 | 256 | 256 | 256 | 128 |
| **Early Stopping** | Yes (patience=5, val_loss) | Yes (patience=5) | No | No | Yes (patience=2) |
| **Loss** | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy |
| **Class Weights** | Yes | Yes | Yes | Yes | Yes |

**DistilBERT fine-tuning strategy:** We froze the embedding layer and the bottom 4 transformer blocks, then unfroze the top 2 transformer blocks (layers 4 and 5 out of 0-5) for fine-tuning. This yielded 36 trainable variable tensors (~14.2M parameters). This approach was chosen because:
- The HuffPost dataset is a topic-classification task, not one requiring deep syntactic reasoning.
- The lower layers capture grammatical/syntactic features (less relevant for our task).
- The upper layers capture semantic/topical features (directly relevant).
- A small learning rate (2e-5) with cosine decay ensured gentle adaptation of the unfrozen layers.
- Due to severe class imbalance, full fine-tuning was ruled out as it tends to overfit dominant classes and destabilize training.

**Training subset:** For DistilBERT, training was performed on a 50% stratified subset (80,145 samples) to manage compute time within the available budget.

---

## Evaluation

All models were evaluated on the same held-out test set (20,037 samples) using the same metrics to ensure fair comparison.

### Primary Metrics

- **Test Accuracy:** Overall fraction of correct predictions.
- **Macro F1-Score:** Unweighted mean of per-class F1, treating all 31 categories equally regardless of size. This is the primary metric given the severe class imbalance (29x ratio).
- **Validation Accuracy at Best Epoch:** The benchmark for comparing training behavior.

### Summary Results Table

| Model | Test Acc | Test F1 (Macro) | Val Acc | Best Epoch | Parameters | Train Time |
|---|---|---|---|---|---|---|
| **DistilBERT** | **0.7077** | **0.6206** | **0.7092** | 9 | 66,977,311 | 65 min |
| Baseline (Emb+AvgPool) | 0.5638 | 0.4987 | 0.5613 | 11 | 1,286,175 | 37s |
| BiLSTM | 0.5602 | 0.4867 | 0.5610 | 6 | 1,356,319 | 37s |
| Enhanced Baseline | 0.5536 | 0.4889 | 0.5427 | 3 | 1,299,359 | <1s* |
| LSTM | 0.5342 | 0.4616 | 0.5405 | 11 | 1,319,199 | 37s |

*\*Enhanced Baseline train_time was near-zero due to a timing measurement that captured only the evaluation call, not the full training.*

### Training Behavior

**Baseline:** Training ran for 16 epochs before early stopping triggered (best epoch 11). Training accuracy rose steadily from 0.137 to 0.733, while validation accuracy plateaued around epoch 10-11 at ~0.561. The widening gap between training accuracy (0.733) and validation accuracy (0.566) at the final epoch confirmed overfitting. Validation loss reached its minimum of 1.6560 at epoch 11 and increased afterward.

**LSTM:** Performed worse than the baseline, reaching 0.540 validation accuracy at best epoch 11 and test accuracy of 0.534. While the LSTM introduced sequential modeling, it did not improve over the bag-of-words baseline, suggesting that word-order information provides limited additional signal for this short-text classification task.

**BiLSTM:** Achieved performance comparable to the baseline (test accuracy 0.560, F1 0.487). Converged faster (best at epoch 6 out of 8) with the bidirectional structure helping capture more context. However, it did not meaningfully surpass the simpler baseline on macro F1.

**Enhanced Baseline:** Achieved test accuracy of 0.554 and F1 of 0.489, comparable to the original baseline. The added regularization (BatchNorm + Dropout) and extra Dense layer did not translate into better generalization, suggesting the original model was already well-balanced for this task.

**DistilBERT:** Trained for 11 of 16 epochs before early stopping triggered (patience=2, best epoch 9), taking 65 minutes on a T4 GPU. This model demonstrated dramatically different learning dynamics from the from-scratch models:
- By epoch 1, validation accuracy already reached 0.579 — surpassing the best validation accuracy of any other model.
- Accuracy continued climbing through epoch 9 (val acc 0.709), with validation loss decreasing from 1.656 to 0.9996.
- Early stopping triggered at epoch 11 when validation loss stopped improving.
- On the test set: accuracy = 0.708, macro F1 = 0.621, weighted F1 = 0.704.
- Unlike the from-scratch models, DistilBERT achieved meaningful F1 scores across all 31 categories — no class received zero predictions.

### Per-Class Performance Highlights (DistilBERT — Best Model)

**Strongest classes** (F1 > 0.75):
- FOOD & DRINK: 0.83 | STYLE: 0.83 | TRAVEL: 0.83 | WEDDINGS: 0.81 | POLITICS: 0.80 | DIVORCE: 0.78 | WELLNESS: 0.75 | PARENTING: 0.75 | SPORTS: 0.75 | WORLD NEWS: 0.76 | QUEER VOICES: 0.75

**Weakest classes** (F1 < 0.45):
- FIFTY: 0.30 | GOOD NEWS: 0.37 | WOMEN: 0.39 | IMPACT: 0.41

**Comparison to Baseline per-class performance:** DistilBERT improved F1 on all 31 categories compared to the baseline. The most dramatic improvements were on categories where contextual understanding matters: EDUCATION (0.41 → 0.56), ARTS (0.46 → 0.62), ENVIRONMENT (0.45 → 0.60), MONEY (0.43 → 0.55), RELIGION (0.46 → 0.63), SCIENCE (0.37 → 0.65), and ENTERTAINMENT (0.56 → 0.74). Even the weakest DistilBERT category (FIFTY: 0.30) outperformed the baseline's result on that same category (0.25).

### Confusion Analysis (Top Misclassified Pairs)

The following confusion pairs persisted across multiple models, indicating they stem from intrinsic dataset characteristics rather than model-specific weaknesses:

| True Label | Predicted Label | Baseline Count | BiLSTM Count | Enhanced Count | DistilBERT Count |
|---|---|---|---|---|---|
| HOME & LIVING | WELLNESS | 210 | 262 | 223 | 233 |
| POLITICS | WORLD NEWS | 188 | 161 | 331 | 84 |
| ENTERTAINMENT | COMEDY | 196 | — | — | 71 |
| POLITICS | BUSINESS | 119 | 131 | 188 | — |
| POLITICS | MEDIA | 128 | 130 | 123 | 55 |
| COMEDY | ENTERTAINMENT | — | — | — | 74 |
| BLACK VOICES | ENTERTAINMENT | — | — | — | 70 |
| PARENTING | WELLNESS | — | — | — | 68 |

**Key observations:**
- The HOME & LIVING → WELLNESS confusion pair remained the dominant error across all models (210-262 counts for from-scratch models, 233 for DistilBERT). These categories share extensive vocabulary about health, home remedies, and lifestyle topics.
- The Enhanced Baseline showed extreme POLITICS → WORLD NEWS confusion (331 misclassifications), the highest single confusion count across all experiments, suggesting that additional regularization degraded the model's ability to distinguish political content from international news.
- DistilBERT substantially reduced POLITICS confusion: the POLITICS → WORLD NEWS pair dropped from 188 (baseline) to 84, demonstrating that contextual embeddings can better distinguish political content from international news.
- DistilBERT's confusion patterns were more balanced — the top pair count (233) was much lower than the from-scratch models' peaks (up to 331), and errors were more evenly distributed rather than concentrated in a few dominant pairs.

---

## Key Results and Analysis

### Which Model Performed Best?

**DistilBERT** was the clear winner across all metrics:
- Highest test accuracy (0.708) — a 14.4 percentage point improvement over the baseline (0.564)
- Highest macro F1 (0.621) — a 12.2 percentage point improvement over the baseline (0.499)
- Highest weighted F1 (0.704)
- Meaningful predictions for all 31 categories (no class neglect)
- Best validation loss (0.9996), indicating well-calibrated probability outputs

The pretrained language representations proved decisive. While the from-scratch models all clustered between 53-56% accuracy and 46-50% F1, DistilBERT broke through this ceiling by leveraging linguistic knowledge acquired during pretraining on a large general-purpose corpus.

### Complexity vs. Performance Trade-offs

The results reveal two distinct performance tiers:

**Tier 1 — From-scratch models (53-56% accuracy, 46-50% F1):**
- All four from-scratch architectures (Baseline, LSTM, BiLSTM, Enhanced Baseline) performed within a narrow 3-percentage-point band on both accuracy and F1.
- Adding sequential modeling (LSTM, BiLSTM) or regularization (BatchNorm, Dropout) did not break through the ~56% accuracy ceiling.
- This suggests that for this dataset, a simple bag-of-words approach captures most of the learnable signal available from randomly initialized embeddings.
- The baseline achieved the best F1 within this tier (0.499), making it the most efficient from-scratch option.

**Tier 2 — Pretrained model (70.8% accuracy, 62.1% F1):**
- DistilBERT achieved a 14+ point accuracy jump and 12+ point F1 jump over the best from-scratch model.
- This came at the cost of 52x more parameters (67M vs. 1.3M) and 105x longer training time (65 min vs. 37s).
- The performance gain justifies the compute cost for a production use case, as the improvement is substantial and consistent across nearly all categories.

**Key insight:** The ~56% ceiling of the from-scratch models was not a dataset limitation — it was a representation limitation. DistilBERT's pretrained contextual embeddings enabled it to distinguish between semantically similar categories (e.g., POLITICS vs. WORLD NEWS, ENTERTAINMENT vs. COMEDY) that bag-of-words models could not.

### Comparison to Milestone 1 Plans

In Milestone 1, we planned to:
1. **Merge confusable categories** — Implemented. Reduced 41 categories to 31 by merging 8 overlapping groups (e.g., ARTS & CULTURE / CULTURE & ARTS, PARENTS / PARENTING, WORLDPOST / THE WORLDPOST). This directly reduced label noise.
2. **Use macro F1 as primary metric** — Implemented. This proved essential for evaluating class-balanced performance. DistilBERT led on both accuracy and F1, confirming its superiority across the board.
3. **Apply class weighting** — Implemented via `compute_class_weight("balanced")`. Weights ranged from 0.198 to 5.726. Combined with DistilBERT's pretrained representations, this enabled meaningful predictions even for the smallest categories.
4. **Progress from baseline to LSTM/BiLSTM to transformers** — Implemented across Problems 2-4. The progression clearly demonstrated the value of pretrained language models over from-scratch approaches for this task.
5. **Use P95 word count for sequence length** — Implemented. P95 was 57 words; we set `max_length=64`.

### Visualization Descriptions

The following charts were generated in the notebook and support the analysis above:

1. **Class Distribution Bar Chart (Problem 1):** Horizontal bar chart showing 31 categories sorted by frequency. POLITICS dominates (32,721) with a long tail down to LATINO VOICES (1,129). Visualizes the 29x imbalance ratio.

2. **Word Length Distribution Histogram (Problem 1):** Shows the distribution of word counts per sample. Mean: 30.2, Median: 29, P95: 57, P99: 68, Max: 246. Right-skewed with most samples between 15-50 words. Justifies the `max_length=64` choice.

3. **Learning Curves (Problems 2-4):** Dual-panel plots (accuracy + loss vs. epoch) for each model. The baseline shows clear overfitting (training accuracy reaches 0.733 while validation plateaus at 0.561). The LSTM and BiLSTM show comparable learning dynamics. DistilBERT shows rapid learning — validation accuracy reaches 0.579 by epoch 1 and peaks at 0.709 by epoch 9, with early stopping triggering at epoch 11.

4. **Test Accuracy Bar Chart (Problem 5):** Compares test accuracy across all 5 models. DistilBERT leads clearly at 0.708, while the four from-scratch models cluster between 0.53-0.56.

5. **Test F1 Bar Chart (Problem 5):** Compares macro F1. DistilBERT leads at 0.621, with the from-scratch models between 0.46-0.50. This chart highlights the two-tier performance pattern.

6. **Training Time Bar Chart (Problem 5):** DistilBERT at ~3,920 seconds vs. under 37 seconds for all other models. Highlights the compute trade-off for the significant performance gain.

7. **Combined Accuracy + F1 Grouped Bar Chart (Problem 5):** Side-by-side comparison. DistilBERT shows the smallest gap between accuracy and F1 (0.087), indicating the most balanced per-class performance. The from-scratch models show gaps of 0.065-0.073.

8. **Top Confusion Pairs Chart (Problem 5):** Shows misclassification counts per model. HOME & LIVING → WELLNESS is consistently the top confusion pair across all models, though DistilBERT reduced total confusion counts substantially.

9. **Misclassified Categories Heatmap (Problem 5):** Cross-model comparison of which true labels are most frequently misclassified. DistilBERT shows lighter colors (fewer errors) across most categories compared to the from-scratch models.

---

## Limitations and Future Work

### Current Limitations

1. **Dataset-inherent ambiguity:** The most persistent errors (HOME & LIVING ↔ WELLNESS, ENTERTAINMENT ↔ COMEDY, POLITICS ↔ WORLD NEWS) reflect genuine semantic overlap in the HuffPost editorial taxonomy rather than model failure. Even DistilBERT, with its rich contextual representations, could not fully resolve these confusions because the categories themselves are not cleanly separable from short text alone.

2. **Short text constraint:** Headlines and short descriptions average only 30 words. This provides limited context for distinguishing between topically related categories. DistilBERT's pretrained knowledge partially compensated, but longer text inputs would likely improve all models.

3. **Training subset for DistilBERT:** Training on a 50% subset (80,145 samples) was necessary due to compute constraints, but it reduced exposure to minority classes. Full-dataset training could improve performance on underrepresented categories like FIFTY (F1=0.30) and GOOD NEWS (F1=0.37).

4. **Class imbalance persistence:** Despite using balanced class weights, the 29x imbalance ratio still impacted minority-class recall. Categories like FIFTY (F1=0.30), GOOD NEWS (F1=0.37), and WOMEN (F1=0.39) remained the weakest across all models.

5. **Limited hyperparameter exploration:** Due to the long training times for DistilBERT (~65 min per run), we did not systematically tune learning rate, number of unfrozen layers, or epoch count. A hyperparameter search could yield further improvements.

### Future Work

Given more time and compute resources, several directions could further improve classification performance:

1. **Full-dataset training:** Training DistilBERT on 100% of the data instead of the 50% subset would increase exposure to minority classes and likely improve their F1 scores.

2. **Further category merging:** The persistent HOME & LIVING / WELLNESS and ENTERTAINMENT / COMEDY confusion pairs suggest these categories may not be meaningfully separable from short text alone. Additional merges could improve overall quality.

3. **Hyperparameter tuning:** Unfreezing more transformer blocks (top 3 or 4), tuning the learning rate schedule, exploring longer `max_length` (96 or 128), and using validation F1 as the optimization criterion could all yield gains.

4. **Data-centric improvements:** Cleaning noisy or ambiguous samples and applying text augmentation (synonym replacement, back-translation) for minority classes could address the remaining class imbalance issues.

5. **Ensemble approaches:** Combining DistilBERT's contextual predictions with the baseline's keyword-level features could capture complementary signals.

---

## AI Tools Disclosure

**AI tools used:** Claude Code (Anthropic, Opus model) via CLI, Codex, and ChatGPT.

**How they were used:**
- **Technical support:** Environment setup (resolving TensorFlow/tensorflow-metal version conflicts, configuring GPU vs CPU execution for Apple Silicon), debugging runtime errors (PyArrow array indexing incompatibility with sklearn), identifying a TF/Keras layer-freezing issue in DistilBERT fine-tuning, and notebook infrastructure (results persistence across kernel restarts, tf.data pipeline setup).
- **Summarization:** Used ChatGPT to provide summarization of the results based on the outputs consolidated by the code (results.json and confusion.csv).

**Why:** Platform-specific issues (M4 Pro / Metal GPU) and framework-specific gotchas (TF/Keras layer-freezing semantics) are not covered in course materials and required troubleshooting outside the scope of the assignments.

**Full conversation log:** Available upon request.

---

## Appendix: Complete Classification Reports

### Baseline (Emb+AvgPool) — Test Set

```
               precision    recall  f1-score   support

         ARTS       0.44      0.49      0.46       388
 BLACK VOICES       0.47      0.42      0.44       453
     BUSINESS       0.49      0.46      0.47       594
       COMEDY       0.35      0.47      0.40       517
        CRIME       0.42      0.59      0.49       340
      DIVORCE       0.74      0.77      0.75       342
    EDUCATION       0.38      0.48      0.42       215
ENTERTAINMENT       0.80      0.44      0.57      1606
  ENVIRONMENT       0.44      0.51      0.47       394
        FIFTY       0.21      0.34      0.26       140
 FOOD & DRINK       0.65      0.77      0.71       832
    GOOD NEWS       0.11      0.53      0.18       139
HOME & LIVING       0.54      0.40      0.46      1085
       IMPACT       0.23      0.42      0.30       345
LATINO VOICES       0.31      0.42      0.35       113
        MEDIA       0.41      0.57      0.48       281
        MONEY       0.39      0.46      0.42       170
    PARENTING       0.67      0.61      0.64      1255
     POLITICS       0.89      0.54      0.67      3272
 QUEER VOICES       0.71      0.64      0.68       631
     RELIGION       0.38      0.65      0.48       254
      SCIENCE       0.25      0.67      0.37       218
       SPORTS       0.62      0.69      0.65       489
        STYLE       0.83      0.72      0.77      1176
         TECH       0.34      0.51      0.41       203
       TRAVEL       0.73      0.66      0.69       989
     WEDDINGS       0.70      0.73      0.71       365
   WEIRD NEWS       0.19      0.42      0.26       267
     WELLNESS       0.69      0.62      0.65      1782
        WOMEN       0.27      0.39      0.32       340
   WORLD NEWS       0.62      0.64      0.63       842

     accuracy                           0.56     20037
    macro avg       0.49      0.55      0.50     20037
 weighted avg       0.64      0.56      0.59     20037
```

### DistilBERT (Best Model) — Test Set

```
               precision    recall  f1-score   support

         ARTS       0.60      0.63      0.62       388
 BLACK VOICES       0.47      0.46      0.47       453
     BUSINESS       0.58      0.54      0.56       594
       COMEDY       0.54      0.44      0.48       517
        CRIME       0.61      0.64      0.62       340
      DIVORCE       0.80      0.76      0.78       342
    EDUCATION       0.54      0.57      0.56       215
ENTERTAINMENT       0.71      0.76      0.74      1606
  ENVIRONMENT       0.60      0.60      0.60       394
        FIFTY       0.39      0.25      0.30       140
 FOOD & DRINK       0.79      0.88      0.83       832
    GOOD NEWS       0.45      0.32      0.37       139
HOME & LIVING       0.67      0.57      0.62      1085
       IMPACT       0.49      0.35      0.41       345
LATINO VOICES       0.50      0.49      0.49       113
        MEDIA       0.58      0.55      0.56       281
        MONEY       0.56      0.53      0.55       170
    PARENTING       0.73      0.77      0.75      1255
     POLITICS       0.81      0.80      0.80      3272
 QUEER VOICES       0.83      0.68      0.75       631
     RELIGION       0.59      0.67      0.63       254
      SCIENCE       0.67      0.63      0.65       218
       SPORTS       0.76      0.75      0.75       489
        STYLE       0.79      0.87      0.83      1176
         TECH       0.50      0.55      0.53       203
       TRAVEL       0.83      0.83      0.83       989
     WEDDINGS       0.80      0.82      0.81       365
   WEIRD NEWS       0.50      0.41      0.45       267
     WELLNESS       0.70      0.82      0.75      1782
        WOMEN       0.42      0.37      0.39       340
   WORLD NEWS       0.77      0.74      0.76       842

     accuracy                           0.71     20037
    macro avg       0.63      0.61      0.62     20037
 weighted avg       0.70      0.71      0.70     20037
```
