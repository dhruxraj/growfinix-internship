"""Task 1 · Spoken Emotion Recognition with a CNN

Plain-script version of the notebook. Install the requirements first:
    pip install -r requirements.txt
"""


# # Task 1 · Spoken Emotion Recognition with a CNN
#
# **Growfinix Machine Learning Internship – Month 2, Task 1**
#
# **Goal:** take speech recordings (`.wav`), convert them into log-mel spectrograms with **Librosa**, and train a
# **Convolutional Neural Network (TensorFlow/Keras)** that predicts whether the speaker's voice conveys
# **anger, happiness or sadness**.
#
# **Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song), speech audio only –
# 24 professional actors, Zenodo record 1188976, licence CC BY-NC-SA 4.0. Only the three required emotions are used.
#
# **Pipeline:** WAV → resample / trim / fixed length → log-mel spectrogram → standardise → CNN → softmax over 3 emotions.
#
# **Evaluation:** speaker-independent split (actors in the test set never appear in training), accuracy,
# per-class precision/recall/F1, macro-F1 and a confusion matrix.
#
# Runs on Google Colab (GPU optional, ~10 minutes) or locally after `pip install -r requirements.txt`.


# Requirements: pip install librosa tensorflow scikit-learn matplotlib


import json
import random
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, f1_score)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
IN_NOTEBOOK = "ipykernel" in sys.modules


def save_and_show(fig, name):
    """Save every figure to outputs/ (for the report and README) and display it in a notebook."""
    fig.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


print("TensorFlow", tf.__version__, "| Librosa", librosa.__version__,
      "| GPU:", tf.config.list_physical_devices("GPU"))


# ## 1. Download the RAVDESS speech audio
# The zip (about 200 MB) is downloaded once into `data/`. If the download is blocked on your network, download
# `Audio_Speech_Actors_01-24.zip` manually from https://zenodo.org/records/1188976 and place it in `data/`.


RAVDESS_URL = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
zip_path = DATA_DIR / "Audio_Speech_Actors_01-24.zip"
audio_dir = DATA_DIR / "ravdess"

if not audio_dir.exists():
    if not zip_path.exists():
        print("Downloading RAVDESS speech audio ...")
        req = urllib.request.Request(RAVDESS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(zip_path, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(audio_dir)

wav_files = sorted(audio_dir.rglob("*.wav"))
print(f"{len(wav_files)} WAV files found")


# ## 2. Select the three required emotions
# RAVDESS file names encode the labels: `03-01-05-01-02-01-12.wav` →
# modality-channel-**emotion**-intensity-statement-repetition-**actor**.
# Emotion codes used here: `05` = angry, `03` = happy, `04` = sad.


EMOTION_CODES = {"05": "anger", "03": "happiness", "04": "sadness"}
CLASSES = ["anger", "happiness", "sadness"]

records = []  # (path, class index, actor id)
for f in wav_files:
    parts = f.stem.split("-")
    if len(parts) == 7 and parts[2] in EMOTION_CODES:
        records.append((f, CLASSES.index(EMOTION_CODES[parts[2]]), int(parts[6])))

print(f"{len(records)} clips selected")
print({CLASSES[k]: v for k, v in sorted(Counter(r[1] for r in records).items())})


# ## 3. Speaker-independent split
# Actors 21–24 form the test set and actors 19–20 the validation set, so the model is always evaluated on voices
# it has never heard. A random split would let the CNN partly recognise the speaker and overstate accuracy.


TEST_ACTORS = {21, 22, 23, 24}
VAL_ACTORS = {19, 20}


def split_of(actor):
    if actor in TEST_ACTORS:
        return "test"
    if actor in VAL_ACTORS:
        return "val"
    return "train"


print(Counter(split_of(a) for _, _, a in records))


# ## 4. Audio preprocessing and log-mel spectrograms
# * load as mono and resample to 22.05 kHz
# * trim leading/trailing silence (`top_db=30`) and pad/crop to 3 s
# * mel power spectrogram (2048-point FFT, hop 512, 128 mel bands) converted to decibels
#
# Each clip becomes a 128 × 130 "image" (mel bands × time frames).


SR = 22050
DURATION = 3.0
N_SAMPLES = int(SR * DURATION)
N_FFT, HOP, N_MELS = 2048, 512, 128


def load_clip(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return librosa.util.fix_length(y, size=N_SAMPLES)


def log_mel(y):
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


# one example spectrogram per emotion
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for cls, ax in enumerate(axes):
    path = next(p for p, c, a in records if c == cls)
    img = librosa.display.specshow(log_mel(load_clip(path)), sr=SR, hop_length=HOP,
                                   x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(f"{CLASSES[cls]} – {path.name}")
fig.colorbar(img, ax=axes, format="%+2.0f dB")
save_and_show(fig, "example_spectrograms.png")


# ## 5. Build the feature arrays (with light augmentation of the training set)
# Each training clip is used twice: once as recorded and once with low-level noise and a small time shift.
# Augmentation is applied to training data only.


rng = np.random.default_rng(SEED)


def augment(y):
    noisy = y + rng.normal(0.0, 0.005, size=y.shape) * (np.max(np.abs(y)) + 1e-9)
    return np.roll(noisy, rng.integers(-SR // 10, SR // 10)).astype(np.float32)


X = {"train": [], "val": [], "test": []}
Y = {"train": [], "val": [], "test": []}
test_paths = []
for i, (path, label, actor) in enumerate(records):
    split = split_of(actor)
    y = load_clip(path)
    X[split].append(log_mel(y))
    Y[split].append(label)
    if split == "train":
        X[split].append(log_mel(augment(y)))
        Y[split].append(label)
    if split == "test":
        test_paths.append(path)
    if (i + 1) % 100 == 0:
        print(f"processed {i + 1}/{len(records)} clips")

X_train, X_val, X_test = (np.stack(X[s])[..., np.newaxis] for s in ("train", "val", "test"))
y_train, y_val, y_test = (np.array(Y[s]) for s in ("train", "val", "test"))

# standardise with statistics from the training set only (no leakage)
MEAN, STD = float(X_train.mean()), float(X_train.std())
X_train, X_val, X_test = [(x - MEAN) / STD for x in (X_train, X_val, X_test)]
print("train", X_train.shape, "val", X_val.shape, "test", X_test.shape)


# ## 6. CNN model
# Three convolution blocks (Conv2D → BatchNorm → MaxPool) learn time–frequency patterns such as harmonic spacing
# (pitch) and energy contours; global average pooling, dropout and a dense layer feed a 3-way softmax.


def build_cnn(input_shape, n_classes):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_cnn(X_train.shape[1:], len(CLASSES))
model.summary()


# ## 7. Training
# Early stopping on the validation loss restores the best weights; the learning rate is halved when progress stalls.


callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
]
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=80, batch_size=32, callbacks=callbacks, verbose=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key in zip(axes, ["loss", "accuracy"]):
    ax.plot(history.history[key], label="training")
    ax.plot(history.history["val_" + key], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.set_title(f"Training curve – {key}")
    ax.legend()
save_and_show(fig, "training_curves.png")


# ## 8. Evaluation on unseen speakers (actors 21–24)


probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)
test_acc = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
report = classification_report(y_test, y_pred, target_names=CLASSES, digits=3)
print(f"Test accuracy: {test_acc:.3f}   Macro-F1: {macro_f1:.3f}\n")
print(report)

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=CLASSES).plot(ax=ax, cmap="Blues",
                                                                                     colorbar=False)
ax.set_title("Confusion matrix – test speakers")
save_and_show(fig, "confusion_matrix.png")

metrics = {
    "dataset": "RAVDESS speech (anger, happiness, sadness)",
    "n_clips": len(records),
    "split": {"train_actors": "1-18 (+1 augmented copy per clip)", "val_actors": "19-20", "test_actors": "21-24"},
    "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
    "epochs_run": len(history.history["loss"]),
    "test_accuracy": round(float(test_acc), 4),
    "test_macro_f1": round(float(macro_f1), 4),
    "classification_report": classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True),
}
(OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
(OUT_DIR / "classification_report.txt").write_text(report)


# ## 9. Save the model and predict the emotion of any WAV file


model.save(str(OUT_DIR / "ser_cnn.keras"))
(OUT_DIR / "normalisation.json").write_text(json.dumps({"mean": MEAN, "std": STD, "classes": CLASSES}))


def predict_emotion(wav_path):
    x = (log_mel(load_clip(wav_path)) - MEAN) / STD
    p = model.predict(x[np.newaxis, ..., np.newaxis], verbose=0)[0]
    return {c: round(float(v), 3) for c, v in zip(CLASSES, p)}


for path in test_paths[:3]:
    true = EMOTION_CODES[path.stem.split("-")[2]]
    print(f"{path.name}  true: {true:9s}  predicted: {predict_emotion(path)}")
