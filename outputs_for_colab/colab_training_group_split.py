import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, Concatenate, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def build_hybrid_model(img_shape, tab_shape, num_classes):
    img_input = Input(shape=img_shape, name="image_input")

    x = Conv2D(16, 3, activation="relu", padding="same")(img_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)

    tab_input = Input(shape=(tab_shape,), name="tabular_input")

    y = Dense(32, activation="relu")(tab_input)
    y = BatchNormalization()(y)
    y = Dropout(0.4)(y)

    y = Dense(16, activation="relu")(y)
    y = Dropout(0.3)(y)

    combined = Concatenate()([x, y])

    z = Dense(64, activation="relu")(combined)
    z = Dropout(0.5)(z)

    output = Dense(num_classes, activation="softmax")(z)

    model = Model(inputs=[img_input, tab_input], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def choose_best_group_split(X, y, groups, n_splits=5):
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    all_classes = set(np.unique(y))
    best_score = -1
    best_split = None

    for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups)):
        train_classes = set(np.unique(y[train_idx]))
        test_classes = set(np.unique(y[test_idx]))

        missing_train = len(all_classes - train_classes)
        missing_test = len(all_classes - test_classes)

        _, train_counts = np.unique(y[train_idx], return_counts=True)
        _, test_counts = np.unique(y[test_idx], return_counts=True)

        score = -100 * (missing_train + missing_test) - abs(len(test_idx) / len(y) - 0.2)

        if score > best_score:
            best_score = score
            best_split = (fold, train_idx, test_idx, missing_train, missing_test)

    return best_split


def save_distribution(path, name, y_values, label_encoder):
    labels, counts = np.unique(y_values, return_counts=True)
    data = {
        str(label_encoder.inverse_transform([label])[0]): int(count)
        for label, count in zip(labels, counts)
    }

    with open(os.path.join(path, f"{name}_class_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    try:
        import google.colab
        BASE_DIR = "/content/drive/MyDrive/aml_project/outputs_for_colab"
    except Exception:
        BASE_DIR = r"C:\Users\ASUS\Desktop\aml_project\outputs_for_colab"

    RESULTS_DIR = os.path.join(BASE_DIR, "results_group_split_v2")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Veriler yükleniyor...")

    X_img = np.load(os.path.join(BASE_DIR, "X_images.npy"))
    X_tab = np.load(os.path.join(BASE_DIR, "X_tabular.npy"))
    y_raw = np.load(os.path.join(BASE_DIR, "y_labels.npy"), allow_pickle=True)
    groups = np.load(os.path.join(BASE_DIR, "groups.npy"), allow_pickle=True)

    print("X_img:", X_img.shape)
    print("X_tab:", X_tab.shape)
    print("y_raw:", y_raw.shape)
    print("groups:", groups.shape)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    num_classes = len(le.classes_)

    with open(os.path.join(RESULTS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("Sınıflar:", le.classes_)

    print("\nEn uygun StratifiedGroupKFold split seçiliyor...")
    fold, train_idx, test_idx, missing_train, missing_test = choose_best_group_split(X_tab, y, groups)

    print(f"Seçilen fold: {fold}")
    print(f"Train örnek sayısı: {len(train_idx)}")
    print(f"Test örnek sayısı: {len(test_idx)}")
    print(f"Train eksik sınıf sayısı: {missing_train}")
    print(f"Test eksik sınıf sayısı: {missing_test}")

    np.savez(
        os.path.join(RESULTS_DIR, "split_indices.npz"),
        train=train_idx,
        test=test_idx
    )

    X_img_train, X_img_test = X_img[train_idx], X_img[test_idx]
    X_tab_train, X_tab_test = X_tab[train_idx], X_tab[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    save_distribution(RESULTS_DIR, "train", y_train, le)
    save_distribution(RESULTS_DIR, "test", y_test, le)

    print("\nTrain class distribution:")
    print(np.unique(y_train, return_counts=True))

    print("\nTest class distribution:")
    print(np.unique(y_test, return_counts=True))

    print("\nTabular veri scale ediliyor...")
    scaler = StandardScaler()
    X_tab_train = scaler.fit_transform(X_tab_train)
    X_tab_test = scaler.transform(X_tab_test)

    with open(os.path.join(RESULTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("\n[1/2] Baseline RandomForest eğitiliyor...")
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=8
    )

    rf.fit(X_tab_train, y_train)
    y_pred_rf = rf.predict(X_tab_test)

    baseline_report = classification_report(
        y_test,
        y_pred_rf,
        target_names=[str(c) for c in le.classes_],
        zero_division=0
    )

    baseline_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred_rf)),
        "macro_f1": float(f1_score(y_test, y_pred_rf, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred_rf, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred_rf, average="weighted", zero_division=0))
    }

    with open(os.path.join(RESULTS_DIR, "metrics_baseline.json"), "w") as f:
        json.dump(baseline_metrics, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "classification_report_baseline.txt"), "w", encoding="utf-8") as f:
        f.write(baseline_report)

    print("Baseline metrics:", baseline_metrics)

    print("\n[2/2] Hybrid CNN + Tabular model kuruluyor...")

    classes = np.unique(y_train)
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weights = {int(cls): float(weight) for cls, weight in zip(classes, cw)}

    print("Class weights:", class_weights)

    model = build_hybrid_model(
        img_shape=X_img_train.shape[1:],
        tab_shape=X_tab_train.shape[1],
        num_classes=num_classes
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        [X_img_train, X_tab_train],
        y_train,
        validation_data=([X_img_test, X_tab_test], y_test),
        epochs=30,
        batch_size=16,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(RESULTS_DIR, "hybrid_model.keras"))

    print("\nHybrid model değerlendiriliyor...")
    y_pred_prob = model.predict([X_img_test, X_tab_test])
    y_pred = np.argmax(y_pred_prob, axis=1)

    target_names = [str(c) for c in le.classes_]

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    hybrid_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "selected_fold": int(fold),
        "missing_train_classes": int(missing_train),
        "missing_test_classes": int(missing_test)
    }

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(hybrid_metrics, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== HYBRID CNN + TABULAR MODEL RESULTS V2 ===\n")
        f.write("Accuracy tek başına güvenilir değildir.\n")
        f.write("Group-based split kullanıldığı için Macro F1, Macro Recall ve per-class Recall önceliklidir.\n\n")
        f.write(report)
        f.write("\n\n=== BASELINE RANDOM FOREST ===\n")
        f.write(json.dumps(baseline_metrics, indent=4))
        f.write("\n\n=== HYBRID METRICS ===\n")
        f.write(json.dumps(hybrid_metrics, indent=4))

    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(RESULTS_DIR, "confusion_matrix.npy"), cm)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_results.png"), dpi=200)
    plt.close()

    print("\nTüm işlemler tamamlandı.")
    print("Sonuç klasörü:", RESULTS_DIR)
    print("Hybrid metrics:", hybrid_metrics)


if __name__ == "__main__":
    main()