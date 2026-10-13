# Task 1 · Spoken Emotion Recognition (Librosa + Keras CNN)

Predicts whether a speaker's voice conveys **anger, happiness or sadness** from a `.wav` recording.

## Dataset
RAVDESS speech audio (24 professional actors), downloaded automatically from Zenodo record 1188976
(`Audio_Speech_Actors_01-24.zip`, ~200 MB). Only the three required emotions are kept: 576 clips, 192 per class.

## Method
1. Load each WAV as mono at 22.05 kHz, trim silence, pad/crop to 3 s (Librosa).
2. Log-mel spectrogram: 2048-point FFT, hop 512, 128 mel bands → a 128 × 130 image per clip.
3. **Speaker-independent split:** actors 1–18 train (each clip plus one noise/shift-augmented copy), 19–20 validation, 21–24 test.
4. CNN: 3 × (Conv2D → BatchNorm → MaxPool) → global average pooling → dropout → Dense(64) → softmax(3).
5. Adam, early stopping on validation loss, learning-rate reduction on plateau.

## Run
Colab: upload `speech_emotion_recognition.ipynb`, choose a GPU runtime, **Run all**.
Locally: `pip install -r requirements.txt`, then open the notebook or run `python speech_emotion_recognition.py`.

## Outputs (`outputs/`)
`example_spectrograms.png`, `training_curves.png`, `confusion_matrix.png`, `classification_report.txt`,
`metrics.json`, `ser_cnn.keras` (trained model) and `normalisation.json`.
`predict_emotion("file.wav")` returns class probabilities for any recording.

## Notes
RAVDESS is acted speech, so performance on natural conversation will be lower. Anger and happiness are both
high-arousal emotions and are the pair most often confused.
