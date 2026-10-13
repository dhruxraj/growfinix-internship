# Growfinix Machine Learning Internship – Month 2 Tasks

Five small machine-learning projects for the Growfinix internship task sheet *Month 2: Deep Learning, Audio & NLP*.
Each task has its own folder with a Jupyter notebook (ready for Google Colab), the same code as a plain Python
script, a `requirements.txt` and a README.

| # | Task | Stack | Dataset | Notebook |
|---|------|-------|---------|----------|
| 1 | Spoken emotion recognition (anger / happiness / sadness) | Python, Librosa, TensorFlow/Keras CNN | RAVDESS speech audio | [`task1_speech_emotion_recognition/speech_emotion_recognition.ipynb`](task1_speech_emotion_recognition/speech_emotion_recognition.ipynb) |
| 2 | Spam & phishing e-mail detection | Python, NLTK, PyTorch LSTM | SpamAssassin public corpus (+ optional phishing CSV) | [`task2_spam_phishing_detection/spam_phishing_detection.ipynb`](task2_spam_phishing_detection/spam_phishing_detection.ipynb) |
| 3 | Assembly-line defect detection (Defective / Perfect) | Python, OpenCV, PyTorch ResNet-18 (transfer learning) | Magnetic Tile Defect dataset | [`task3_assembly_line_defect_detection/defect_detection.ipynb`](task3_assembly_line_defect_detection/defect_detection.ipynb) |
| 4 | Collaborative-filtering movie recommender | Python, Scikit-Surprise (SVD), Pandas | MovieLens 100K | [`task4_movie_recommender/movie_recommender.ipynb`](task4_movie_recommender/movie_recommender.ipynb) |
| 5 | Automated machine learning | Python, H2O AutoML (PyCaret version included) | Breast Cancer Wisconsin (scikit-learn) | [`task5_automl/automl_h2o.ipynb`](task5_automl/automl_h2o.ipynb) |

Every notebook downloads its own data, so nothing has to be prepared by hand.

## Run on Google Colab (recommended)

1. Open <https://colab.research.google.com> → **File → Upload notebook** and choose the task's `.ipynb` file.
2. For tasks 1–3, select **Runtime → Change runtime type → T4 GPU** (they also run on CPU, just more slowly).
3. **Runtime → Run all.** Each notebook takes roughly 5–15 minutes.
4. Results are written to an `outputs/` folder in the Colab file browser: `metrics.json`, a classification report
   and PNG figures (confusion matrices, training curves, leaderboards). Download them from the Files panel.

Task 5's PyCaret notebook needs Python 3.9–3.11, so run it locally; the H2O notebook works on Colab.

## Run locally

```bash
cd task1_speech_emotion_recognition        # or any other task folder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook                           # open the .ipynb, or run:  python speech_emotion_recognition.py
```

H2O (task 5) also needs Java 11 or newer installed on your computer.

## Outputs to use in the internship report

| Task | Key figures written to `outputs/` | Key numbers in `outputs/metrics.json` |
|------|-----------------------------------|---------------------------------------|
| 1 | `example_spectrograms.png`, `training_curves.png`, `confusion_matrix.png` | test accuracy, macro-F1, per-class precision/recall |
| 2 | `training_history.png`, `evaluation.png` (confusion matrix + precision–recall curve) | precision, recall, F1, ROC-AUC, false positives/negatives |
| 3 | `dataset_examples.png`, `training_curves.png`, `confusion_matrix.png`, `example_predictions.png` | accuracy, precision/recall/F1 for *Defective* |
| 4 | `eda.png`, `algorithm_comparison.png`, `error_distribution.png`, `top10_user_196.csv` | RMSE, MAE, precision@10, recall@10, best SVD parameters |
| 5 | `eda.png`, `holdout_evaluation.png`, `variable_importance.png`, `leaderboard.csv` | selected model, cross-validated AUC, hold-out AUC/accuracy/F1 |

## Datasets and licences

* **RAVDESS** – S. R. Livingstone and F. A. Russo, *PLoS ONE* 13(5): e0196391, 2018. Zenodo record 1188976, CC BY-NC-SA 4.0.
* **SpamAssassin public corpus** – Apache SpamAssassin project, <https://spamassassin.apache.org/old/publiccorpus/>.
* **Magnetic Tile Defect dataset** – Y. Huang et al., "Surface Defect Saliency of Magnetic Tile", 2018;
  <https://github.com/abin24/Magnetic-tile-defect-datasets.>
* **MovieLens 100K** – F. M. Harper and J. A. Konstan, *ACM TiiS* 5(4), 2015; GroupLens Research, research use.
* **Breast Cancer Wisconsin (Diagnostic)** – UCI Machine Learning Repository, distributed with scikit-learn.

The datasets are downloaded at run time and are not stored in this repository (see `.gitignore`).
