"""Лабораторная работа: перенос обучения на cats_vs_dogs.

Зависимости (Python 3.12, TensorFlow CPU 2.20, TFDS 4.9.9):
    pip install tensorflow-cpu==2.20.0 tensorflow-datasets==4.9.9 "protobuf>=5.28.3,<7" matplotlib importlib-resources

Полный эксперимент (VGG16 и ResNet50, изображения 150 и 224 пикселя):
    python Task_01.py
Сокращённый эксперимент на настоящих изображениях для проверки на CPU:
    python Task_01.py --epochs 2 --fine-tune-epochs 1 --batch-size 16 \
        --max-train-samples 512 --max-val-samples 128 --max-test-samples 512

При первом запуске загружаются датасет (~787 МиБ) и веса ImageNet.
В results/ сохраняются метрики, графики, лучшая модель, матрица ошибок,
подписанные примеры распознавания и выводы по полученным значениям.
Импорт модуля не запускает скачивание или обучение.

Источники:
https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
https://www.tensorflow.org/guide/keras/transfer_learning
"""

import argparse
import gc
import json
import os
from pathlib import Path
import time

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(PROJECT_DIR / ".cache" / "keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")  # Сохранение графиков работает без графической оболочки.
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_NAME = "cats_vs_dogs"
# У cats_vs_dogs нет отдельного test: делим единственный split train.
SPLITS = ("train[:70%]", "train[70%:85%]", "train[85%:]")
AUTOTUNE = tf.data.AUTOTUNE


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Значение должно быть больше нуля.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("vgg16", "resnet50"),
        default=["vgg16", "resnet50"],
    )
    parser.add_argument("--sizes", nargs="+", type=positive_int, default=[150, 224])
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--fine-tune-epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--max-train-samples", type=positive_int)
    parser.add_argument("--max-val-samples", type=positive_int)
    parser.add_argument("--max-test-samples", type=positive_int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=positive_int, default=8)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_DIR / ".cache" / "tensorflow_datasets"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)
    if min(args.sizes) < 32:
        parser.error(
            "Для VGG16 и ResNet50 размер изображения должен быть не меньше 32."
        )
    if len(set(args.sizes)) != len(args.sizes) or len(set(args.architectures)) != len(
        args.architectures
    ):
        parser.error("Архитектуры и размеры не должны повторяться.")
    return args


def load_data(args):
    """Создаём три непересекающиеся выборки; ограничения применяются после разбиения."""
    datasets, info = tfds.load(
        DATASET_NAME,
        split=list(SPLITS),
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
        data_dir=str(args.data_dir),
    )
    limits = (args.max_train_samples, args.max_val_samples, args.max_test_samples)
    datasets = [
        ds.take(limit) if limit is not None else ds
        for ds, limit in zip(datasets, limits)
    ]
    counts = [int(tf.data.experimental.cardinality(ds).numpy()) for ds in datasets]
    if any(count <= 0 for count in counts):
        raise ValueError(
            "Обучающая, валидационная и тестовая выборки должны быть непустыми."
        )
    class_names = info.features["label"].names
    print("Классы:", class_names, flush=True)
    print("Число изображений (train / validation / test):", *counts, flush=True)
    return datasets, class_names, counts


def prepare_data(raw_dataset, size, batch_size, seed, training=False):
    """Батчи RGB в [0, 255]; исходные изображения разных размеров обрабатываются по одному."""
    if training:
        count = int(tf.data.experimental.cardinality(raw_dataset).numpy())
        raw_dataset = raw_dataset.shuffle(
            min(count, 1000), seed=seed, reshuffle_each_iteration=True
        )

    def resize(image, label):
        return tf.image.resize(image, (size, size), antialias=True), label

    options = tf.data.Options()
    options.experimental_deterministic = True
    return (
        raw_dataset.with_options(options)
        .map(resize, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


def build_model(architecture, size, num_classes, seed, weights="imagenet"):
    """Одинаковая классификационная голова для сравнения двух архитектур."""
    constructors = {
        "vgg16": tf.keras.applications.VGG16,
        "resnet50": tf.keras.applications.ResNet50,
    }
    preprocessors = {
        "vgg16": tf.keras.applications.vgg16.preprocess_input,
        "resnet50": tf.keras.applications.resnet50.preprocess_input,
    }
    base = constructors[architecture](
        include_top=False, weights=weights, input_shape=(size, size, 3)
    )
    base.trainable = False
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.1, seed=seed + 2),
        ],
        name="augmentation",
    )
    inputs = tf.keras.Input(shape=(size, size, 3), name="rgb_image")
    x = augmentation(inputs)
    # Препроцессинг ImageNet: RGB -> BGR, вычитание среднего.
    x = preprocessors[architecture](x)
    # BatchNormalization остаётся в inference-режиме и при дообучении.
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3, seed=seed + 3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="classes")(
        x
    )
    return tf.keras.Model(inputs, outputs, name=f"{architecture}_{size}"), base


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_last_block(base, architecture):
    """Дообучаем последний свёрточный блок, оставляя BatchNormalization замороженным."""
    base.trainable = True
    prefix = "block5_" if architecture == "vgg16" else "conv5_"
    for layer in base.layers:
        layer.trainable = layer.name.startswith(prefix) and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )


def callbacks(checkpoint, initial_best=None):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            initial_value_threshold=initial_best,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=2
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def plot_history(history, head_epochs, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["loss"]) + 1)
    for axis, metric, title in zip(axes, ("accuracy", "loss"), ("Точность", "Потери")):
        axis.plot(epochs, history[metric], label="Обучение")
        axis.plot(epochs, history[f"val_{metric}"], label="Валидация")
        axis.axvline(
            head_epochs + 0.5, color="gray", linestyle="--", label="Начало дообучения"
        )
        axis.set(xlabel="Эпоха", title=title)
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(architecture, size, raw_train, raw_val, num_classes, args):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.seed)
    name = f"{architecture}_{size}"
    directory = args.output_dir / name
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "best.weights.h5"
    train = prepare_data(raw_train, size, args.batch_size, args.seed, training=True)
    validation = prepare_data(raw_val, size, args.batch_size, args.seed)
    model, base = build_model(architecture, size, num_classes, args.seed)
    print(f"\nЭксперимент {name}: обучение классификатора", flush=True)
    started = time.perf_counter()
    compile_model(model, 1e-3)
    head = model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=callbacks(checkpoint),
        shuffle=False,
        verbose=2,
    )
    model.load_weights(checkpoint)
    head_metrics = model.evaluate(validation, return_dict=True, verbose=0)

    print(f"{name}: дообучение последнего блока", flush=True)
    unfreeze_last_block(base, architecture)
    compile_model(model, 1e-5)
    fine = model.fit(
        train,
        validation_data=validation,
        epochs=args.fine_tune_epochs,
        callbacks=callbacks(checkpoint, head_metrics["accuracy"]),
        shuffle=False,
        verbose=2,
    )
    elapsed = time.perf_counter() - started
    # Если дообучение ухудшило accuracy, остаются лучшие веса первого этапа.
    model.load_weights(checkpoint)
    metrics = model.evaluate(validation, return_dict=True, verbose=0)
    history = {
        key: [float(x) for x in head.history[key] + fine.history[key]]
        for key in ("accuracy", "val_accuracy", "loss", "val_loss")
    }
    result = {
        "name": name,
        "architecture": architecture,
        "image_size": size,
        "head_val_accuracy": float(head_metrics["accuracy"]),
        "val_accuracy": float(metrics["accuracy"]),
        "val_loss": float(metrics["loss"]),
        "training_seconds": elapsed,
        "head_epochs": len(head.history["loss"]),
        "fine_tune_epochs": len(fine.history["loss"]),
        "parameters": model.count_params(),
        "checkpoint": str(checkpoint.resolve()),
    }
    save_json(directory / "history.json", history)
    save_json(directory / "metrics.json", result)
    plot_history(history, result["head_epochs"], directory / "training_history.png")
    print(
        f"{name}: validation accuracy = {metrics['accuracy']:.4f}, время = {elapsed:.1f} с",
        flush=True,
    )
    del model, base, train, validation
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def evaluate_test(model, dataset, num_classes, example_count=5):
    """Один проход по всему тесту; в памяти метрики и максимум десять изображений."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    examples = {"correct": [], "incorrect": []}
    loss_sum, total = 0.0, 0
    for images, labels in dataset:
        probabilities = model(images, training=False).numpy()
        truth = labels.numpy()
        predicted = np.argmax(probabilities, axis=1)
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels, probabilities
        ).numpy()
        loss_sum += float(np.sum(losses))
        total += len(truth)
        np.add.at(confusion, (truth, predicted), 1)
        for index, (actual, guess) in enumerate(zip(truth, predicted)):
            group = "correct" if actual == guess else "incorrect"
            if len(examples[group]) < example_count:
                examples[group].append(
                    {
                        "image": np.clip(images[index].numpy(), 0, 255).astype(
                            np.uint8
                        ),
                        "true_label": int(actual),
                        "predicted_label": int(guess),
                        "confidence": float(probabilities[index, guess]),
                    }
                )
    if total == 0:
        raise ValueError("Тестовая выборка пуста.")
    support = confusion.sum(axis=1)
    precision = np.divide(
        np.diag(confusion),
        confusion.sum(axis=0),
        out=np.zeros(num_classes, dtype=float),
        where=confusion.sum(axis=0) != 0,
    )
    recall = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros(num_classes, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=float),
        where=precision + recall != 0,
    )
    return {
        "accuracy": float(np.trace(confusion) / total),
        "loss": loss_sum / total,
        "macro_f1": float(f1.mean()),
        "samples": total,
        "correct": int(np.trace(confusion)),
        "incorrect": int(total - np.trace(confusion)),
        "confusion_matrix": confusion.tolist(),
        "per_class": [
            {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        ],
    }, examples


def plot_examples(examples, class_names, title, path):
    fig, axes = plt.subplots(1, 5, figsize=(17, 5), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, example in zip(axes.flat, examples):
        axis.imshow(example["image"])
        axis.set_title(
            f"Истина: {class_names[example['true_label']]}\n"
            f"Прогноз: {class_names[example['predicted_label']]}\n"
            f"Уверенность: {example['confidence']:.1%}",
            fontsize=10,
        )
    fig.suptitle(f"{title} (показано {len(examples)} из 5)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if len(examples) < 5:
        print(
            f"{title}: во всём тесте найдено только {len(examples)} примеров.",
            flush=True,
        )


def plot_confusion(matrix, class_names, path):
    matrix = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(6, 5))
    display = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(display, ax=axis)
    axis.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Предсказанный класс",
        ylabel="Истинный класс",
        title="Матрица ошибок на тесте",
    )
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > matrix.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_conclusions(results, best, test, counts, args):
    lines = [
        "Лабораторная работа: перенос обучения на cats_vs_dogs.",
        f"Выборки train / validation / test: {counts[0]} / {counts[1]} / {counts[2]}.",
        "Датасет заменён на cats_vs_dogs, выходной слой содержит 2 класса (cat, dog).",
        "Единственный split train разделён на непересекающиеся части 70% / 15% / 15%.",
        "Модель и лучшие веса выбраны по валидации; тест используется после выбора.",
        "",
        "Сравнение на одинаковых данных:",
        "Модель | Размер | Accuracy головы | Лучшая val accuracy | Val loss | Обучение, с",
    ]
    for result in results:
        lines.append(
            f"{result['architecture']} | {result['image_size']} | "
            f"{result['head_val_accuracy']:.4f} | {result['val_accuracy']:.4f} | "
            f"{result['val_loss']:.4f} | {result['training_seconds']:.1f}"
        )
    lines.extend(
        [
            "",
            f"Выбрана модель {best['name']} (val accuracy {best['val_accuracy']:.4f}).",
            f"Независимый тест: accuracy = {test['accuracy']:.4f}, loss = {test['loss']:.4f}, "
            f"macro F1 = {test['macro_f1']:.4f}.",
            f"Правильных ответов: {test['correct']}; ошибок: {test['incorrect']}.",
        ]
    )
    for size in args.sizes:
        pair = {r["architecture"]: r for r in results if r["image_size"] == size}
        if "vgg16" in pair and "resnet50" in pair:
            delta = 100 * (
                pair["resnet50"]["val_accuracy"] - pair["vgg16"]["val_accuracy"]
            )
            lines.append(
                f"При {size}x{size} разница val accuracy ResNet50 − VGG16: {delta:+.2f} п.п."
            )
    for architecture in args.architectures:
        runs = sorted(
            (r for r in results if r["architecture"] == architecture),
            key=lambda r: r["image_size"],
        )
        if len(runs) > 1:
            small, large = runs[0], runs[-1]
            delta = 100 * (large["val_accuracy"] - small["val_accuracy"])
            lines.append(
                f"Для {architecture} увеличение {small['image_size']} -> {large['image_size']} "
                f"изменило val accuracy на {delta:+.2f} п.п.; время обучения "
                f"{small['training_seconds']:.1f} -> {large['training_seconds']:.1f} с."
            )
    gain = 100 * (best["val_accuracy"] - best["head_val_accuracy"])
    lines.extend(
        [
            f"Прирост val accuracy выбранной модели после дообучения: {gain:+.2f} п.п. "
            "Веса с меньшей точностью не заменяют лучший результат.",
            "Использованы веса ImageNet, аугментация только при обучении, GlobalAveragePooling, "
            "Dropout, ранняя остановка и малый learning rate при дообучении последнего блока.",
            "Матрица ошибок и подписанные изображения позволяют проверить характер ошибок.",
            "Выводы относятся к данному разбиению и seed; один запуск не доказывает общее превосходство архитектуры.",
        ]
    )
    if any(
        value is not None
        for value in (
            args.max_train_samples,
            args.max_val_samples,
            args.max_test_samples,
        )
    ):
        lines.append(
            "Число примеров ограничено: это сокращённый эксперимент. "
            "Для окончательной оценки повторите его без --max-*-samples."
        )
    if test["incorrect"] < 5:
        lines.append(
            "В тесте меньше пяти ошибок: показаны все реальные ошибки без подмены изображений."
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    datasets, class_names, counts = load_data(args)
    raw_train, raw_val, raw_test = datasets
    results = []
    for architecture in args.architectures:
        for size in args.sizes:
            results.append(
                run_experiment(
                    architecture, size, raw_train, raw_val, len(class_names), args
                )
            )

    # Тест не участвует в ранней остановке или выборе архитектуры/размера.
    best = max(
        results, key=lambda result: (result["val_accuracy"], -result["val_loss"])
    )
    model, _ = build_model(
        best["architecture"],
        best["image_size"],
        len(class_names),
        args.seed,
        weights=None,
    )
    model.load_weights(best["checkpoint"])
    test_data = prepare_data(raw_test, best["image_size"], args.batch_size, args.seed)
    print(f"\nНезависимый тест лучшей модели: {best['name']}", flush=True)
    test, examples = evaluate_test(model, test_data, len(class_names))
    test["class_names"] = class_names
    model.save(args.output_dir / "best_model.keras")
    plot_examples(
        examples["correct"],
        class_names,
        "Правильно распознанные",
        args.output_dir / "correct_examples.png",
    )
    plot_examples(
        examples["incorrect"],
        class_names,
        "Неправильно распознанные",
        args.output_dir / "incorrect_examples.png",
    )
    plot_confusion(
        test["confusion_matrix"], class_names, args.output_dir / "confusion_matrix.png"
    )
    example_labels = {
        group: [
            {key: value for key, value in item.items() if key != "image"}
            for item in items
        ]
        for group, items in examples.items()
    }
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    save_json(
        args.output_dir / "results.json",
        {
            "dataset": DATASET_NAME,
            "splits": SPLITS,
            "counts": counts,
            "config": configuration,
            "versions": {
                "tensorflow": tf.__version__,
                "tensorflow_datasets": tfds.__version__,
                "numpy": np.__version__,
            },
            "experiments": results,
            "selected_model": best["name"],
            "test": test,
            "examples": example_labels,
        },
    )
    conclusions = make_conclusions(results, best, test, counts, args)
    (args.output_dir / "conclusions.txt").write_text(conclusions, encoding="utf-8")
    print("\n" + conclusions, flush=True)
    print("Результаты сохранены в:", args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
