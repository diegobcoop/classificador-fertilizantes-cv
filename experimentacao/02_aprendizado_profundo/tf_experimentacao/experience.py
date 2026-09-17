import os
import time
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models, mixed_precision, backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    resnet,
    mobilenet_v2,
    mobilenet_v3,
    efficientnet,
    efficientnet_v2,
    convnext,
    densenet,
    xception,
    inception_v3,
)
from tensorflow.keras import applications as keras_apps

from core.config import CONFIG
from core import reporting as rp
from core import utils


# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU RTX detectada e configurada.")
    except RuntimeError as e:
        print(e)


# ---------------------------------------------------------------------------
# Registro de modelos e pré-processamento
# ---------------------------------------------------------------------------

MODEL_FACTORY = {
    'resnet50':        resnet.ResNet50,
    'xception':        xception.Xception,
    'efficientnet_b0': efficientnet.EfficientNetB0,
}

PREPROCESS_MAP = {
    'resnet50':           resnet.preprocess_input,
    'mobilenet_v2':       mobilenet_v2.preprocess_input,
    'inception_v3':       inception_v3.preprocess_input,
    'xception':           xception.preprocess_input,
    'densenet121':        densenet.preprocess_input,
    'efficientnet_b0':    efficientnet.preprocess_input,
    'efficientnet_b3':    efficientnet.preprocess_input,
    'efficientnet_v2_s':  efficientnet_v2.preprocess_input,
    'convnext_tiny':      convnext.preprocess_input,
    'convnext_small':     convnext.preprocess_input,
    'mobilenet_v3_large': keras_apps.mobilenet_v3.preprocess_input,
}


# ---------------------------------------------------------------------------
# Classes e funções auxiliares
# ---------------------------------------------------------------------------

class MonteCarloDropout(layers.Dropout):
    """Dropout que permanece ativo em inferência para estimativa de incerteza."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def get_preprocessing_fn(model_name):
    return PREPROCESS_MAP.get(model_name)


def get_backbone(model_name, input_shape, weights='imagenet'):
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Modelo '{model_name}' não suportado.")

    is_weight_file = isinstance(weights, str) and weights.endswith(('.h5', '.keras'))
    base_model = MODEL_FACTORY[model_name](
        input_shape=input_shape,
        include_top=False,
        weights=None if is_weight_file else weights,
    )
    if is_weight_file:
        base_model.load_weights(weights, by_name=True, skip_mismatch=True)
    return base_model


def build_model(base_model, num_classes):
    return models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        MonteCarloDropout(CONFIG['DROPOUT']),
        layers.Dense(num_classes, activation='softmax', dtype='float32'),
    ])


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_acc'),
        ],
    )


def build_augmented_generators(train_paths, train_labels, val_paths, val_labels,
                                class_names, preprocess_fn):
    """Cria geradores com data augmentation a partir de listas de paths."""
    rescale = 1. / 255 if preprocess_fn is None else None

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=rescale,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    def to_dataframe(paths, labels):
        return pd.DataFrame({'filename': paths, 'class': [class_names[l] for l in labels]})

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=to_dataframe(train_paths, train_labels),
        x_col='filename', y_col='class',
        target_size=CONFIG['IMAGE_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
    )
    val_gen = val_datagen.flow_from_dataframe(
        dataframe=to_dataframe(val_paths, val_labels),
        x_col='filename', y_col='class',
        target_size=CONFIG['IMAGE_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        shuffle=False,
    )
    return train_gen, val_gen


def build_tf_dataset(paths, labels, num_classes, preprocess_fn, shuffle):
    """Cria um tf.data.Dataset a partir de listas de paths e labels."""
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, CONFIG['IMAGE_SIZE'])
        if preprocess_fn:
            img = tf.numpy_function(preprocess_fn, [img], tf.float32)
            img.set_shape(CONFIG['IMAGE_SIZE'] + (3,))
        else:
            img = img / 255.0
        return img, tf.one_hot(label, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=42)
    return ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) \
             .batch(CONFIG['BATCH_SIZE']) \
             .prefetch(tf.data.AUTOTUNE)


def load_all_image_paths(directory):
    """Carrega paths e labels inteiros de um diretório organizado por classes."""
    class_names = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    all_paths, all_labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(valid_exts):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label_idx)
    return all_paths, all_labels, class_names


def get_mc_predictions(model, data_gen, iterations=20):
    """Roda inferência múltiplas vezes com Dropout ativo e retorna média e desvio."""
    all_preds = np.array([model.predict(data_gen, verbose=0) for _ in range(iterations)])
    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)


def measure_fps(model, val_gen, data_aug):
    """Mede imagens por segundo processadas pelo modelo."""
    img_count, inf_start = 0, time.time()
    if data_aug:
        for _ in range(min(5, len(val_gen))):
            img, _ = next(val_gen)
            model.predict(img, verbose=0)
            img_count += img.shape[0]
    else:
        for img, _ in val_gen.take(5):
            model.predict(img, verbose=0)
            img_count += img.shape[0]
    duration = time.time() - inf_start
    return img_count / duration if duration > 0 else 0


def extract_y_true(val_gen, data_aug):
    """Extrai labels verdadeiros do gerador de validação."""
    if data_aug:
        val_gen.reset()
        return val_gen.classes
    y_true = []
    for _, labels in val_gen:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    return y_true


def run_evaluation(model, val_gen, data_aug, model_tag, class_names, output_path):
    """Avalia o modelo: FPS, Monte Carlo, matriz de confusão e relatório."""
    fps = measure_fps(model, val_gen, data_aug)

    print("Calculando predições com Monte Carlo Dropout...")
    mean_preds, std_preds = get_mc_predictions(model, val_gen, iterations=30)
    y_pred = np.argmax(mean_preds, axis=1)
    avg_incerteza = np.mean([std_preds[i][y_pred[i]] for i in range(len(y_pred))])
    print(f"⚠️ Incerteza Média do Modelo: {avg_incerteza:.4f}")

    y_true = extract_y_true(val_gen, data_aug)
    if data_aug:
        model.predict(val_gen)  # necessário após reset para realinhar o gerador

    rp.plot_confusion_matrix(model_tag, y_true, y_pred, class_names, output_path)

    file_report = f'relatorio_classificacao_{model_tag}.csv'
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    utils.save_classification_report(report, file_report, output_path)
    rp.plot_report_metrics(model_tag.split('_', 1)[-1], output_path, file_report)
    rp.plot_summary_metrics(model_tag.split('_', 1)[-1], output_path, file_report)

    return fps, y_pred


def train_two_phase(model, base_model, train_gen, val_gen):
    """Executa as duas fases de treino: feature extraction e fine-tuning."""
    early_stop = EarlyStopping(
        monitor='val_loss', patience=CONFIG['PATIENCE'], restore_best_weights=True
    )

    print("Iniciando Fase 1: Feature Extraction...")
    compile_model(model, CONFIG['INITIAL_LR'])
    history_f1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=CONFIG['INITIAL_EPOCHS'], callbacks=[early_stop],
    )

    print(f"Iniciando Fase 2: Fine-Tuning ({CONFIG['UNFREEZE_LAYERS']} camadas liberadas)...")
    base_model.trainable = True
    for layer in base_model.layers[:-CONFIG['UNFREEZE_LAYERS']]:
        layer.trainable = False
    compile_model(model, CONFIG['FINE_TUNE_LR'])

    start = time.time()
    history_f2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=CONFIG['FINE_TUNE_EPOCHS'], callbacks=[early_stop],
    )
    train_time = time.time() - start

    return history_f1, history_f2, train_time


# ---------------------------------------------------------------------------
# Persistência de resultados
# ---------------------------------------------------------------------------

BENCHMARK_PATH = os.path.join(CONFIG['OUTPUT_MODEL'], 'benchmark_rtx4060.json')


def load_results():
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_result(result):
    results = load_results()
    results[result['Modelo']] = result
    with open(BENCHMARK_PATH, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Experimento sem K-Fold
# ---------------------------------------------------------------------------

def run_experiment(model_name, data_aug):
    backend.clear_session()
    print(f"\n{'='*40}\nPROCESSANDO: {model_name}\n{'='*40}")

    if model_name in modelos_executados:
        print(f"[SKIP] {model_name} já processado.")
        return modelos_executados[model_name]

    folder_tag = os.path.basename(CONFIG['OUTPUT_MODEL'])
    model_tag  = f"{folder_tag}_{model_name}"
    output_path = os.path.join(CONFIG['OUTPUT_MODEL'], model_name)
    os.makedirs(output_path, exist_ok=True)

    preprocess_fn = get_preprocessing_fn(model_name)
    input_shape   = CONFIG['IMAGE_SIZE'] + (3,)

    # Geradores
    if data_aug:
        all_paths, all_labels, class_names = load_all_image_paths(CONFIG['TRAIN_DIR'])
        val_paths, val_labels, _ = load_all_image_paths(CONFIG['VAL_DIR'])
        num_classes = len(class_names)
        train_gen, val_gen = build_augmented_generators(
            all_paths, all_labels, val_paths, val_labels, class_names, preprocess_fn
        )
    else:
        train_gen = tf.keras.utils.image_dataset_from_directory(
            CONFIG['TRAIN_DIR'],
            image_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            label_mode='categorical',
        )
        val_gen = tf.keras.utils.image_dataset_from_directory(
            CONFIG['VAL_DIR'],
            image_size=CONFIG['IMAGE_SIZE'],
            batch_size=CONFIG['BATCH_SIZE'],
            label_mode='categorical',
        )
        class_names = train_gen.class_names
        num_classes  = len(class_names)

    # Modelo
    base_model = get_backbone(model_name, input_shape)
    base_model.trainable = False
    model = build_model(base_model, num_classes)

    history_f1, history_f2, train_time = train_two_phase(model, base_model, train_gen, val_gen)

    # Salvar modelo e histórico
    model.save(os.path.join(output_path, f"modelo_{model_tag}_final.keras"))
    print(f"✅ Modelo salvo em: {output_path}")

    df_hist = utils.save_training_history(history_f1, history_f2, model_tag, output_path)
    rp.plot_learning_curves(df_hist, model_tag, output_path)

    # Avaliação
    fps, y_pred = run_evaluation(model, val_gen, data_aug, model_tag, class_names, output_path)

    result = {
        'Modelo':          model_name,
        'Acurácia':        round(max(history_f2.history['val_accuracy']), 4),
        'FPS (RTX 4060)':  round(fps, 2),
        'VRAM_Estimada':   'Alta' if 'convnext' in model_name else 'Baixa',
        'Total_Params':    f"{model.count_params():,}",
        'Tempo_FT_Seg':    round(train_time, 2),
    }
    save_result(result)
    return result


# ---------------------------------------------------------------------------
# Experimento com K-Fold
# ---------------------------------------------------------------------------

def run_experiment_kfold(model_name, data_aug, n_splits=5):
    backend.clear_session()
    print(f"\n{'='*40}\nPROCESSANDO: {model_name}\n{'='*40}")

    if model_name in modelos_executados:
        print(f"[SKIP] {model_name} já processado.")
        return modelos_executados[model_name]

    folder_tag  = os.path.basename(CONFIG['OUTPUT_MODEL'])
    model_tag   = f"{folder_tag}_{model_name}"
    output_path = os.path.join(CONFIG['OUTPUT_MODEL'], model_name)
    os.makedirs(output_path, exist_ok=True)

    preprocess_fn = get_preprocessing_fn(model_name)
    input_shape   = CONFIG['IMAGE_SIZE'] + (3,)

    all_paths, all_labels, class_names = load_all_image_paths(CONFIG['TRAIN_DIR'])
    num_classes = len(class_names)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_results  = []
    best_val_acc  = -1
    best_model    = None
    best_history  = (None, None)
    best_val_gen  = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_paths, all_labels)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        backend.clear_session()

        train_paths  = [all_paths[i]  for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths    = [all_paths[i]  for i in val_idx]
        val_labels   = [all_labels[i] for i in val_idx]

        if data_aug:
            train_gen, val_gen = build_augmented_generators(
                train_paths, train_labels, val_paths, val_labels, class_names, preprocess_fn
            )
        else:
            train_gen = build_tf_dataset(train_paths, train_labels, num_classes, preprocess_fn, shuffle=True)
            val_gen   = build_tf_dataset(val_paths,   val_labels,   num_classes, preprocess_fn, shuffle=False)

        base_model = get_backbone(model_name, input_shape)
        base_model.trainable = False
        model = build_model(base_model, num_classes)

        history_f1, history_f2, train_time = train_two_phase(model, base_model, train_gen, val_gen)

        fold_val_acc = max(history_f2.history['val_accuracy'])
        fold_results.append({'fold': fold + 1, 'val_accuracy': fold_val_acc, 'train_time': train_time})
        print(f"Fold {fold + 1} val_accuracy: {fold_val_acc:.4f}")

        if fold_val_acc > best_val_acc:
            best_val_acc = fold_val_acc
            best_model   = model
            best_history = (history_f1, history_f2)
            best_val_gen = val_gen

    # Salvar modelo e histórico do melhor fold
    best_model.save(os.path.join(output_path, f"modelo_{model_tag}_final.keras"))
    print(f"✅ Modelo salvo em: {output_path}")

    history_f1, history_f2 = best_history
    df_hist = utils.save_training_history(history_f1, history_f2, model_tag, output_path)
    rp.plot_learning_curves(df_hist, model_tag, output_path)

    # Resultados por fold
    df_folds = pd.DataFrame(fold_results)
    df_folds.to_csv(os.path.join(output_path, f'kfold_results_{model_name}.csv'), index=False)
    print(df_folds.to_string(index=False))

    # Avaliação do melhor fold
    fps, y_pred = run_evaluation(best_model, best_val_gen, data_aug, model_tag, class_names, output_path)

    accs = [r['val_accuracy'] for r in fold_results]
    result = {
        'Modelo':                model_name,
        'Acurácia':              round(best_val_acc, 4),
        'Acurácia_Media_KFold':  round(float(np.mean(accs)), 4),
        'Acurácia_Std_KFold':    round(float(np.std(accs)), 4),
        'FPS (RTX 4060)':        round(fps, 2),
        'VRAM_Estimada':         'Alta' if 'convnext' in model_name else 'Baixa',
        'Total_Params':          f"{best_model.count_params():,}",
        'Tempo_FT_Seg':          round(sum(r['train_time'] for r in fold_results), 2),
    }
    save_result(result)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    start_process = time.time()
    modelos_executados = load_results()

    os.makedirs(CONFIG['OUTPUT_MODEL'], exist_ok=True)

    modelos  = list(MODEL_FACTORY.keys())
    data_aug = CONFIG['APPLY_DATA_AUGMENTATION']

    [run_experiment_kfold(m, data_aug) for m in modelos]

    results = list(load_results().values())
    rp.plot_comparison_results(results)
    rp.gerar_graficos_avancados(results)
    utils.save_results_csv(results, CONFIG['OUTPUT_MODEL'], 'benchmark_rtx4060.csv')

    print(f'Tempo total: {time.time() - start_process:.2f}s')