# Changes applied to the Technical Report

Source: `DX703 Final Project - Technical Report.docx` (teammates' draft, Apr 24 22:35)
Target: `DX703 Final Project - Technical Report (updated).docx` (my edits, Apr 24 23:44)

All edits reflect the final full-data Kaggle run: **Test Acc 0.6609, Macro F1 0.6004**, 24 scheduled epochs with early stopping at epoch 11 (best at 9), trained on full 160,290 samples.

---

## 1. Section B.2.3 Training Strategy

**Replaced** the body paragraph to include explicit hyperparameters in prose (AdamW, weight decay 0.01, 2e-5 peak LR with cosine schedule + 10% warmup, batch 128, max_len 64, sparse categorical cross-entropy, early stopping patience=2). Kept the existing freeze-strategy rationale.

## 2. Table 1 (Training Configurations)

**Max Epochs: 16 → 24**

## 3. Section B.2.4 Performance Validation

**Rewrote** the paragraph for the full-data run:
- Dropped "50% stratified subset (80,145 samples)"
- Added: full training set (160,290 samples), 24 scheduled epochs, ES at epoch 11 (best at 9), ~111 min on T4 GPU
- Kept the "full fine-tuning ruled out" and "avoided data leakage" language

## 4. Section B.3.1 Quantitative Results

**Rewrote** with new numbers (0.661 accuracy / 0.600 macro-F1 / 0.673 weighted-F1) and added an explicit justification for macro-F1 as the primary metric (class imbalance, why accuracy alone misleads).

## 5. Section B.3.2 Interpretation of Results

**Rewrote** the two paragraphs:
- New top confusion pairs from the current run (HOME & LIVING → WELLNESS 160, POLITICS → MEDIA 156, POLITICS → WORLD NEWS 133, ENTERTAINMENT → COMEDY 128, POLITICS → BUSINESS 111)
- Added observation linking POLITICS precision 0.92 / recall 0.57 pattern to the class-weight / imbalance dynamic

## 6. Section B.3.3 Summary of Final Model Performance

**Rewrote** with honest full-data numbers. Frames DistilBERT's ~10 pp improvement over from-scratch baselines, discusses strengths / weaknesses, and explicitly states the 111 min / 67M param cost is justified for this task.

## 7. Section B.4.1 Limitations

**Rewrote** the 5-limitation paragraph:
- Limitation 3 changed from "50% subset was necessary" (no longer true) to "capacity limitation: partial fine-tuning under-utilized the full dataset (0.661 vs 0.708 on 50%); both EPOCHS=16 and EPOCHS=24 converged to ~0.66"
- Other 4 limitations kept in spirit

## 8. Section B.4.2 Potential Improvements

**Restructured** from list-style ("To address X... To improve Y...") into **two narrative paragraphs**: data-side improvements (category consolidation, longer inputs, minority augmentation) and model-side improvements (unfreeze top 3-4 blocks, hyperparameter search).

## 9. Section B.5 AI Use Disclosure

**Added** disclosure content (was previously empty). Covers:
- Claude Code for code review + technical fixes (env setup, GPU config, debugging, notebook structure)
- ChatGPT for rephrasing sentences/paragraphs for clarity
- Explicit statement that design, modeling, hyperparameters, and results interpretation are team-authored
- CDS policy compliance

## 10. Table 2 (Appendix B — Classification Report)

**Replaced all 31 data rows** with the new per-class precision/recall/F1/support values from the full-data run.

---

## How to merge

1. Open the latest teammate version (`DX703 Final Project - Technical Report.docx`)
2. For each change above, locate the section and paste in the updated text from the `(updated)` copy
3. The Table 1 change is one cell. The Table 2 change is all 31 rows.
4. Figures 1–4 were not touched — if teammates regenerated them for the full-data run, keep theirs; otherwise, regenerate from the new notebook outputs (`results/` folder: per_class_f1.png, confusion_matrix_normalized.png, top_confused_pairs.png, support_vs_f1.png, model_comparison.png).
