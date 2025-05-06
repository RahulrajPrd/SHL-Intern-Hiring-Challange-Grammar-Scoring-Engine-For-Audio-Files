# Grammar Scoring Engine for Voice Samples
## SHL Research Intern Kaggle Competition Submission
**Author**: Rahul Raj Parida  
**Date**: May 6, 2025  
**Objective**: Build a model to predict continuous grammar scores (0–5) for 45–60-second WAV audio files based on the MOS Likert Grammar Scores rubric.  
**Environment**: Google Colab (Python 3.8+, CPU or GPU)  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Implementation Summary](#implementation-summary)
4. [Evaluation Results](#evaluation-results)
5. [Challenges and Solutions](#challenges-and-solutions)
6. [Future Improvements](#future-improvements)
7. [How to Run](#how-to-run)
8. [Submission Instructions](#submission-instructions)
9. [References](#references)

---

## Project Overview
This project is my submission for the SHL Research Intern Kaggle competition. The goal was to create a grammar scoring engine that predicts continuous grammar scores (0–5) for spoken English audio clips, based on the MOS Likert Grammar Scores rubric. Using 444 training and 195 test audio files, I built a pipeline in Google Colab to process audio, extract features, and predict scores. This README serves as my project report, summarizing the work, results, and challenges faced under a 16-hour deadline.

---

## Dataset Description
The dataset, provided via Kaggle, includes:
- **Training Set**:
  - 444 WAV audio files (45–60 seconds) in `/content/Dataset/audios/train`.
  - `train.csv`: Contains `filename` and `label` (continuous scores, e.g., 0.5, 1.5, 2.0).
- **Test Set**:
  - 195 WAV audio files in `/content/Dataset/audios/test`.
  - `test.csv`: Contains `filename` (no scores).
- **Sample Submission**:
  - `sample_submission.csv`: Template with `filename` and `score` columns.

The dataset required careful handling due to variable audio quality and the need for fast processing.

---

## Implementation Summary
The pipeline processes audio into grammar score predictions:
- **Transcription**: Used Whisper’s `tiny` model to convert audio to text quickly (~40–60 minutes for 639 files).
- **Feature Extraction**: Extracted grammar features (`error_count`, `error_rate`) with LanguageTool and text features (`num_words`, `num_unique_words`, `avg_word_length`, `num_sentences`, `avg_sentence_length`) to improve performance.
- **Modeling**: Trained an XGBoost regressor to predict continuous scores (0–5).
- **Evaluation**: Measured training RMSE, validation RMSE, and Pearson correlation, with visualizations in `notebook.ipynb`.

All code is in `notebook.ipynb`, which installs dependencies (e.g., `whisper`, `language-tool-python`, `xgboost`) and handles Java 17 setup for LanguageTool.

---

## Evaluation Results
Here’s how the model performed:
- **Full Training RMSE**: [Insert full_train_rmse, e.g., 0.4567]
- **Training RMSE (Split)**: [Insert train_rmse, e.g., 0.4321]
- **Validation MSE**: [Insert val_mse, e.g., 0.2456]
- **Validation RMSE**: [Insert val_rmse, e.g., 0.4956]
- **Validation Pearson Correlation**: [Insert val_pearson, e.g., 0.6789]
- **Kaggle Score**: [Pending submission]

Visualizations in `.ipynb` file include:
- **Scatter Plot**: Predicted vs. actual scores on validation set.
- **Histogram**: Distribution of predicted and actual scores.

The model captures grammar trends but is limited by transcription accuracy and simplified features. Continuous scores improved performance over earlier integer attempts.

---

## Challenges and Solutions
1. **Low Initial Score (0.012)**:
   - **Challenge**: Early Kaggle submission scored 0.012 due to limited features and incorrect integer outputs.
   - **Solution**: Added text features (`num_words`, etc.) and switched to continuous scores with `XGBRegressor`.
2. **Score Format Confusion**:
   - **Challenge**: Assumed integer scores (1–5) from rubric, but `train['label']` was continuous, causing a `ValueError` with `XGBClassifier`.
   - **Solution**: Used `XGBRegressor` to match continuous ground truth, aligning with Pearson correlation metric.
3. **Transcription Speed**:
   - **Challenge**: Google Speech Recognition API was too slow (>2 hours for 444 files).
   - **Solution**: Switched to Whisper `tiny`, transcribing in ~40–60 minutes. Loaded existing transcriptions from CSVs.
4. **Java Compatibility**:
   - **Challenge**: LanguageTool needed Java 17, not Colab’s default Java 11.
   - **Solution**: Installed OpenJDK 17 and set `JAVA_HOME`.
5. **Time Constraint**:
   - **Challenge**: 16-hour deadline limited feature complexity.
   - **Solution**: Focused on fast tools (Whisper Tiny, LanguageTool) and saved intermediate CSVs.

---

## Future Improvements
- Use Whisper `base` or `medium` for better transcription accuracy.
- Add advanced features (e.g., SpaCy for syntax, BERT for contextual grammar).
- Tune XGBoost hyperparameters (e.g., `learning_rate`, `max_depth`).
- Implement ensemble models for robustness.
- Process all audio fully to avoid empty transcriptions.

