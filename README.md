# Grammar Scoring Engine for Voice Samples
## SHL Research Intern Kaggle Competition Submission
**Author**: Rahul Raj Parida
**Date**: May 6, 2025  
**Objective**: Build a model to predict grammar scores (0–5) for 45–60-second WAV audio files based on the MOS Likert Grammar Scores rubric.  
**Environment**: Google Colab (Python 3.8+, CPU or GPU)  
---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Implementation Details](#implementation-details)
   - [Setup and Dependencies](#setup-and-dependencies)
   - [Speech-to-Text Transcription](#speech-to-text-transcription)
   - [Feature Extraction](#feature-extraction)
   - [Modeling and Evaluation](#modeling-and-evaluation)
   - [Test Predictions and Submission](#test-predictions-and-submission)
5. [Evaluation Results](#evaluation-results)
6. [Challenges and Solutions](#challenges-and-solutions)
7. [Future Improvements](#future-improvements)
8. [How to Run](#how-to-run)
9. [Submission Instructions](#submission-instructions)
10. [References](#references)

---

## Project Overview
This project develops a grammar scoring engine for the SHL Research Intern Kaggle competition. The goal is to predict grammar scores (0–5) for 45–60-second WAV audio files based on their spoken content, using the MOS Likert Grammar Scores rubric. The pipeline transcribes audio to text, extracts grammar-related features, and trains a regression model to predict scores. The solution was built under a tight 16-hour deadline, prioritizing speed and reliability while addressing computational and data challenges.

---

## Dataset Description
The dataset, provided via Kaggle, includes:
- **Training Set**:
  - 444 WAV audio files (45–60 seconds each) in `/content/Dataset/audios/train`.
  - `train.csv`: Contains `filename` (audio file names) and `label` (grammar scores, 0–5).
- **Test Set**:
  - 195 WAV audio files in `/content/Dataset/audios/test`.
  - `test.csv`: Contains `filename` (no scores).
- **Sample Submission**:
  - `sample_submission.csv`: Template with `filename` and `score` columns for Kaggle submission.

The dataset posed challenges due to variable audio quality and the need for accurate transcription within time constraints.

---

## Pipeline Architecture
The pipeline transforms raw audio into grammar score predictions:

1. **Transcription**: Convert audio to text using Whisper’s `tiny` model for speed.
2. **Feature Extraction**: Extract grammar features (error count, error rate) using LanguageTool.
3. **Modeling**: Train an XGBoost regressor to predict continuous scores (0–5).

---

## Implementation Details
### Setup and Dependencies
- **Environment**: Google Colab with Python 3.8+, CPU or GPU.
- **Libraries**:
  - `whisper`: For fast audio transcription.
  - `language-tool-python`: For grammar feature extraction.
  - `xgboost`: For regression modeling.
  - `pandas`, `numpy`, `sklearn`, `matplotlib`: For data processing and visualization.
- **Java 17**: Installed for LanguageTool compatibility using OpenJDK 17.
- **Key Fix**: Resolved Java version mismatch (Colab’s default Java 11 vs. required Java 17).

### Speech-to-Text Transcription
- **Initial Approach**: Used `speech_recognition` with Google Speech Recognition API, but it was too slow (>2 hours for 444 training files due to network latency and API limits).
- **Solution**: Switched to OpenAI’s Whisper `tiny` model, which transcribes offline and processes ~639 files (444 train + 195 test) in ~40–60 minutes on Colab’s CPU.
- **Implementation**:
  - Loaded existing transcriptions from `train_with_transcriptions.csv` and `test_with_transcriptions.csv` to save time.
  - Handled failed transcriptions by assigning empty strings.
  - Saved transcriptions to CSVs to avoid recomputation.
- **Outcome**: Achieved reliable transcriptions, though `tiny` model accuracy is lower than larger models.

### Feature Extraction
- **Tool**: LanguageTool for grammar analysis.
- **Features**:
  - `error_count`: Number of grammar errors in transcribed text.
  - `error_rate`: Errors per sentence (normalized by sentence count).
- **Process**:
  - Applied LanguageTool to each transcription.
  - Assigned zero values for empty transcriptions.
  - Saved features to `train_features.csv` and `test_features.csv`.
- **Rationale**: Focused on grammar-specific features to align with the MOS rubric, omitting SpaCy for speed.

### Modeling and Evaluation
- **Model**: XGBoost regressor (`n_estimators=100`, `random_state=42`).
- **Data**:
  - Features: `train_features` (error_count, error_rate).
  - Target: `train['label']` (grammar scores).
  - Split: 80% training, 20% validation (random_state=42).
- **Evaluation Metrics**:
  - Mean Squared Error (MSE).
  - Root Mean Squared Error (RMSE).
- **Visualization**: Scatter plot of predicted vs. actual scores.

### Test Predictions and Submission
- **Prediction**: Applied trained XGBoost model to `test_features`.
- **Post-processing**: Clipped predictions to [0, 5] per rubric.
- **Output**: Generated `submission.csv` with `filename` and `score` columns, matching `sample_submission.csv` format.

---

## Evaluation Results
- **Validation Performance**:
  - **MSE**: [Insert MSE from output, e.g., 0.2456]
  - **RMSE**: [Insert RMSE from output, e.g., 0.4956]
- **Analysis**:
  - The model captures general grammar trends but may miss nuances due to simplified features and partial transcriptions.
  - RMSE indicates reasonable accuracy for a baseline under time constraints.
- **Kaggle Leaderboard**:
  - [Insert score if submitted, e.g., "Achieved MSE of X.XX on public leaderboard"] or [Pending submission].

---

## Challenges and Solutions
1. **Transcription Speed**:
   - **Challenge**: Google Speech Recognition API took >2 hours for 444 training files due to network latency and API limits.
   - **Solution**: Switched to Whisper `tiny`, reducing transcription to ~40–60 minutes. Loaded existing transcriptions to avoid reprocessing.
2. **Java Compatibility**:
   - **Challenge**: LanguageTool required Java 17, but Colab defaulted to Java 11.
   - **Solution**: Installed OpenJDK 17 and set `JAVA_HOME`, enabling seamless feature extraction.
3. **Data Issue (KeyError)**:
   - **Challenge**: Code expected `score` column, but `train.csv` used `label`, causing `KeyError: 'score'`.
   - **Solution**: Identified `label` as the score column via `train.columns` and updated the code.
4. **Time Constraint**:
   - **Challenge**: 16-hour deadline limited transcription and feature complexity.
   - **Solution**: Prioritized fast tools (Whisper Tiny, LanguageTool) and simplified features, ensuring a complete submission.

---

## Future Improvements
- **Transcription**: Use larger Whisper models (e.g., `base` or `small`) or fine-tune on domain-specific audio for higher accuracy.
- **Features**: Incorporate advanced NLP features:
  - SpaCy: Sentence length, syntactic complexity.
  - BERT-based models: Contextual grammar scoring.
- **Modeling**: Experiment with ensemble models (e.g., Random Forest, Gradient Boosting) or hyperparameter tuning for XGBoost.
- **Data**: Process all audio files completely with more time to avoid empty transcriptions.
- **Evaluation**: Conduct cross-validation to ensure robustness.

---

## How to Run
### Prerequisites
- **Environment**: Google Colab (free tier, CPU or GPU).
- **Data**:
  - Place in `/content/Dataset/`:
    - `audios/train/` (444 WAV files)
    - `audios/test/` (195 WAV files)
    - `train.csv`, `test.csv`, `sample_submission.csv`
  - Or download via Kaggle API:
    ```bash
    !kaggle competitions download -c <competition-name>
    !unzip <competition-name>.zip -d /content/Dataset
    ```
- **Dependencies**: Installed automatically by the notebook.

### Steps
1. **Open Colab**:
   - Create a new notebook and upload `notebook.ipynb` (or copy code).
   - Set runtime to CPU or GPU (GPU slightly faster for Whisper).
2. **Run Notebook**:
   - Execute cells sequentially.
   - Key outputs:
     - Transcription: Loads `train_with_transcriptions.csv` and `test_with_transcriptions.csv` or transcribes (~40–60 minutes if needed).
     - Features: Generates `train_features.csv` and `test_features.csv`.
     - Model: Trains XGBoost and outputs MSE/RMSE.
     - Submission: Creates `submission.csv`.
3. **Monitor**:
   - Check transcription progress (errors logged but handled).
   - Verify submission format:
     ```python
     print(pd.read_csv('submission.csv').head())
     ```
4. **Save**:
   - Download `submission.csv` and `notebook.ipynb`.
   - Save features and transcriptions to avoid recomputation.

### Example Code
The full pipeline is in `notebook.ipynb`. Key snippet for modeling:
```python
X = train_features
y = train['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
test_predictions = xgb_model.predict(test_features)
test_predictions = np.clip(test_predictions, 0, 5)
submission = pd.DataFrame({'filename': test['filename'], 'score': test_predictions})
submission.to_csv('submission.csv', index=False)
