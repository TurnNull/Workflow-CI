import argparse
import json
import os
import random
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from mlflow.models import infer_signature


# ============================================================
# Konfigurasi reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Argumen command line
# ============================================================

def parse_args():
    """Membaca argumen dari MLflow Project."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="PlantVillage_preprocessing",
        help="Path dataset hasil preprocessing."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Jumlah epoch training."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Ukuran batch."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=160,
        help="Ukuran input gambar."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate optimizer."
    )

    return parser.parse_args()


# ============================================================
# Fungsi dataset
# ============================================================

def load_labels(data_dir: Path) -> list:
    """Membaca daftar label dari labels.txt."""
    labels_path = data_dir / "labels.txt"

    if not labels_path.exists():
        raise FileNotFoundError(f"File labels.txt tidak ditemukan: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file.readlines() if line.strip()]

    return labels


def load_metadata(data_dir: Path) -> dict:
    """Membaca metadata dataset jika tersedia."""
    metadata_path = data_dir / "dataset_metadata.json"

    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def build_image_datasets(data_dir: Path, image_size: int, batch_size: int):
    """Membuat dataset TensorFlow dari folder train, val, dan test."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Folder train tidak ditemukan: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Folder val tidak ditemukan: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Folder test tidak ditemukan: {test_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds


# ============================================================
# Fungsi model
# ============================================================

def build_model(num_classes: int, image_size: int, learning_rate: float):
    """
    Membangun model CNN ringan untuk kebutuhan CI.

    Model dibuat ringan agar workflow GitHub Actions tidak terlalu lama.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(image_size, image_size, 3), name="input_gambar"),
            tf.keras.layers.Rescaling(1.0 / 255, name="normalisasi"),

            tf.keras.layers.Conv2D(16, 3, activation="relu", name="conv_1"),
            tf.keras.layers.MaxPooling2D(name="pool_1"),

            tf.keras.layers.Conv2D(32, 3, activation="relu", name="conv_2"),
            tf.keras.layers.MaxPooling2D(name="pool_2"),

            tf.keras.layers.Conv2D(64, 3, activation="relu", name="conv_3"),
            tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling"),

            tf.keras.layers.Dropout(0.3, name="dropout"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output_klasifikasi"),
        ],
        name="plantvillage_ci_cnn"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Fungsi artefak
# ============================================================

def save_model_summary(model, output_path: Path) -> None:
    """Menyimpan ringkasan arsitektur model ke file TXT."""
    with open(output_path, "w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))


def save_metrics(metrics: dict, output_path: Path) -> None:
    """Menyimpan metrik evaluasi ke file JSON."""
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_run_information(run_id: str, model_uri: str, output_dir: Path) -> None:
    """Menyimpan Run ID dan Model URI untuk kebutuhan workflow CI."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "run_id.txt", "w", encoding="utf-8") as file:
        file.write(run_id)

    with open(output_dir / "model_uri.txt", "w", encoding="utf-8") as file:
        file.write(model_uri)


# ============================================================
# Pipeline training
# ============================================================

def main():
    """Menjalankan training model melalui MLflow Project."""
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path("outputs").resolve()
    artifact_dir = Path("training_artifacts").resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Jangan memakai mlflow.set_experiment() di dalam script MLflow Project.
    # Nama experiment akan diberikan dari command mlflow run melalui --experiment-name.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print("[INFO] Tracking URI:", mlflow.get_tracking_uri())

    labels = load_labels(data_dir)
    metadata = load_metadata(data_dir)
    num_classes = len(labels)

    print("[INFO] Dataset      :", data_dir)
    print("[INFO] Jumlah kelas :", num_classes)
    print("[INFO] Epoch        :", args.epochs)
    print("[INFO] Batch size   :", args.batch_size)
    print("[INFO] Image size   :", args.image_size)

    train_ds, val_ds, test_ds = build_image_datasets(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    model = build_model(
        num_classes=num_classes,
        image_size=args.image_size,
        learning_rate=args.learning_rate
    )

    # Saat dijalankan dengan mlflow run, MLflow sudah membuat run aktif.
    # mlflow.start_run() di sini akan menggunakan run tersebut.
    with mlflow.start_run(run_name="ci_training_plantvillage") as run:
        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        mlflow.log_param("model_name", "CNN_Ringan_CI")
        mlflow.log_param("dataset_name", "PlantVillage")
        mlflow.log_param("dataset_source", "PlantVillage_preprocessing")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("image_size", args.image_size)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("seed", SEED)

        if metadata:
            mlflow.log_dict(metadata, "dataset/dataset_metadata.json")

        labels_path = data_dir / "labels.txt"
        if labels_path.exists():
            mlflow.log_artifact(str(labels_path), artifact_path="dataset")

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            verbose=1
        )

        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))

        for metric_name, values in history.history.items():
            for epoch, value in enumerate(values, start=1):
                mlflow.log_metric(metric_name, float(value), step=epoch)

        metrics = {
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "last_train_accuracy": float(history.history["accuracy"][-1]),
            "last_val_accuracy": float(history.history["val_accuracy"][-1]),
        }

        model_summary_path = artifact_dir / "model_summary.txt"
        metrics_path = artifact_dir / "metrics.json"

        save_model_summary(model, model_summary_path)
        save_metrics(metrics, metrics_path)

        mlflow.log_artifact(str(model_summary_path), artifact_path="model_info")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

        sample_images, _ = next(iter(train_ds.take(1)))
        input_example = sample_images[:1].numpy().astype(np.float32)

        mlflow.tensorflow.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        save_run_information(
            run_id=run_id,
            model_uri=model_uri,
            output_dir=output_dir
        )

        print("[INFO] Run ID   :", run_id)
        print("[INFO] Model URI:", model_uri)
        print("[INFO] Training CI selesai.")


if __name__ == "__main__":
    main()