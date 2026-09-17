import os
import tensorflow as tf
import numpy as np
import time
import json
import pandas as pd

from core.config import CONFIG
from core import reporting as rp
from core import utils

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import layers, models, mixed_precision, backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Novo import
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


class MonteCarloDropout(layers.Dropout):
    def call(self, inputs, training=None):
        # Ignora o flag 'training' do Keras e força True
        return super().call(inputs, training=True)

# --- OTIMIZAÇÃO PARA RTX 4060 ---
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU RTX detectada e configurada.")
    except RuntimeError as e:
        print(e)

# Separado os 3 melhores
MODEL_FACTORY = {
    # Modelos Convolucionais Clássicos
    'resnet50':           resnet.ResNet50,
    'mobilenet_v2':       mobilenet_v2.MobileNetV2,
    'inception_v3':       inception_v3.InceptionV3,
    'xception':           xception.Xception,
    'densenet121':        densenet.DenseNet121,

    # # Família EfficientNet
    'efficientnet_b0':    efficientnet.EfficientNetB0,
    'efficientnet_b3':    efficientnet.EfficientNetB3,
    'efficientnet_v2_s':  efficientnet_v2.EfficientNetV2S,

    # # Família MobileNet V3 e ConvNeXt
    'mobilenet_v3_large': keras_apps.MobileNetV3Large,
    'convnext_tiny':      convnext.ConvNeXtTiny,
    'convnext_small':     convnext.ConvNeXtSmall,
}

# --- FUNÇÕES DE APOIO ---

def get_preprocessing_fn(model_name):
    """Retorna a função de pré-processamento oficial de cada módulo."""
    mapping = {
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
    return mapping.get(model_name)

def load_all_image_paths(directory):
    """Carrega todos os caminhos e labels do diretório de treino."""
    class_names = sorted(os.listdir(directory))
    all_paths, all_labels = [], []
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_paths.append(os.path.join(class_dir, fname))
                all_labels.append(label_idx)
    return all_paths, all_labels, class_names


def load_results():
    if os.path.exists(os.path.join(CONFIG['OUTPUT_MODEL'], 'benchmark_rtx4060.json')):
        with open(os.path.join(CONFIG['OUTPUT_MODEL'], 'benchmark_rtx4060.json'), 'r') as f:
            return json.load(f)
    return {}

def save_result(result):
    results = load_results()
    results[result['Modelo']] = result
    with open(os.path.join(CONFIG['OUTPUT_MODEL'], 'benchmark_rtx4060.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def get_backbone(model_name, input_shape, weights='imagenet'):
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Modelo {model_name} não suportado.")
    
    is_path = isinstance(weights, str) and weights.endswith(('.h5', '.keras'))
    w = None if is_path else weights    
    
    base_model = MODEL_FACTORY[model_name](
        input_shape=input_shape, include_top=False, weights=w
    )
    
    if is_path:
        base_model.load_weights(weights, by_name=True, skip_mismatch=True)
        
    return base_model

def get_mc_predictions(model, data_gen, iterations=20):
    # Roda a mesma imagem várias vezes e calcula a média e o desvio padrão.
    all_preds = []
    for _ in range(iterations):
        # A cada iteração, o Dropout apaga neurônios diferentes
        preds = model.predict(data_gen, verbose=0)
        all_preds.append(preds)
    
    all_preds = np.array(all_preds)
    mean_results = np.mean(all_preds, axis=0) # Predição média (mais estável)
    std_results = np.std(all_preds, axis=0)   # Incerteza (Desvio Padrão)
    
    return mean_results, std_results


def build_dataset(directory, image_size, batch_size, preprocess_fn, training, seed):

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=training,
        seed=seed if training else None
    )
    
    class_names = ds.class_names  # 🔥 GUARDA AQUI
    AUTOTUNE = tf.data.AUTOTUNE

    # 🔥 Data Augmentation (equivalente ao ImageDataGenerator)
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.14),        # ~25 graus
        layers.RandomTranslation(0.2, 0.2),
        layers.RandomZoom(0.2),
    ])

    def process(x, y):
        x = tf.cast(x, tf.float32)

        # ⚠️ Ordem correta
        if training:
            x = data_augmentation(x, training=True)

        if preprocess_fn:
            x = preprocess_fn(x)
        else:
            x = x / 255.0

        return x, y

    ds = ds.map(process, num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)

    return ds, class_names

def measure_fps(model, ds, steps=10, warmup=3):
    # 🔥 Warmup
    for x, _ in ds.take(warmup):
        model.predict(x, verbose=0)

    start = time.time()
    count = 0

    for x, _ in ds.take(steps):
        model.predict(x, verbose=0)
        count += x.shape[0]

    duration = time.time() - start
    return count / duration if duration > 0 else 0

def run_experiment(model_name, data_aug, seed):    
    backend.clear_session()

    tf.random.set_seed(seed)
    np.random.seed(seed)

    key = f"{model_name}_{seed}"
    print(f"\n{'='*40}\nPROCESSANDO: {key}\n{'='*40}")
    
    
    if key in modelos_executados:
        print(f"[SKIP] {key} já processado, pulando...")
        return modelos_executados[key]

    folder_name_output = os.path.basename(CONFIG['OUTPUT_MODEL'])
    output_path = os.path.join(CONFIG['OUTPUT_MODEL'], key)
    print(f'PATH: {output_path}')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"Diretório criado: {output_path}")
    
    # 1. Configuração dos Geradores (Data Augmentation Agressivo)
    preprocess_fn = get_preprocessing_fn(model_name)

    train_gen, class_names = build_dataset(
        CONFIG['TRAIN_DIR'],
        CONFIG['IMAGE_SIZE'],
        CONFIG['BATCH_SIZE'],
        preprocess_fn,
        training=True,
        seed=seed
    )

    val_gen, _ = build_dataset(
        CONFIG['VAL_DIR'],
        CONFIG['IMAGE_SIZE'],
        CONFIG['BATCH_SIZE'],
        preprocess_fn,
        training=False,
        seed=seed
    )

    # Classes
    num_classes = len(class_names)
    input_shape = CONFIG['IMAGE_SIZE'] + (3,)

    # 2. Construção (Removido Sequential de Augmentation pois o Gerador já faz)
    base_model = get_backbone(model_name, input_shape)
    base_model.trainable = False 
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        # layers.Dropout(CONFIG['DROPOUT']),
        # layers.Dropout(CONFIG['DROPOUT'])(None, training=True),
        MonteCarloDropout(CONFIG['DROPOUT']), # Dropout para Monte Carlo
        layers.Dense(num_classes, activation='softmax', dtype='float32')
    ])
    
    metrics_list = [
        'accuracy', 
        tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_acc')
    ]    

    # 3. Compilação (Mudança para categorical_crossentropy)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['INITIAL_LR']),
                loss='categorical_crossentropy', 
                #   metrics=['accuracy']
                metrics=metrics_list)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=CONFIG['PATIENCE'], restore_best_weights=True)

    # 4. Fase 1: Feature Extraction
    print("Iniciando Fase 1...")
    history_f1 = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG['INITIAL_EPOCHS'], callbacks=[early_stop])

    # 5. Fase 2: Fine-Tuning
    print(f"Liberando {CONFIG['UNFREEZE_LAYERS']} camadas...")
    base_model.trainable = True
    for layer in base_model.layers[:-CONFIG['UNFREEZE_LAYERS']]:
        layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=CONFIG['FINE_TUNE_LR']),
                loss='categorical_crossentropy', 
                #   metrics=['accuracy']
                metrics=metrics_list)

    start_train = time.time()
    history_f2 = model.fit(train_gen, validation_data=val_gen, epochs=CONFIG['FINE_TUNE_EPOCHS'], callbacks=[early_stop])
    train_time = time.time() - start_train

    # 6. Salvamento
        
    model_filename = f"modelo_{folder_name_output+'_'+model_name}_final.keras"
    model_full_path = os.path.join(output_path, model_filename)
    model.save(model_full_path)
    print(f"✅ MODELO SALVO EM: {model_full_path}")

    # Salvar e plotar
    df_hist = utils.save_training_history(history_f1, history_f2, folder_name_output+'_'+model_name, output_path)
    rp.plot_learning_curves(df_hist, folder_name_output+'_'+model_name, output_path)


    # 7. Benchmark de Inferência (FPS)
    fps = measure_fps(model, val_gen)
    

    # Executando a predição estocastica (Monte Carlo Dropout) para obter incerteza
    print("Calculando predições com Monte Carlo Dropout...")
    mean_preds, std_preds = get_mc_predictions(model, val_gen, iterations=30)
    y_pred = np.argmax(mean_preds, axis=1) # Predição final é a média das predições
    incerteza = [std_preds[i][y_pred[i]] for i in range(len(y_pred))] # Incerteza associada à classe predita
    avg_incerteza = np.mean(incerteza)
    print(f"⚠️ Incerteza Média do Modelo: {avg_incerteza:.4f}")
    

    # 8. Matriz de Confusão e Relatório
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)

    y_true = []
    for _, labels in val_gen:
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    rp.plot_confusion_matrix(folder_name_output+'_'+model_name, y_true, y_pred, class_names, output_path)
    
    file_report =  f'relatorio_classificacao_{folder_name_output+'_'+model_name}.csv';
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    utils.save_classification_report(report, file_report, output_path)

    rp.plot_report_metrics(model_name, output_path, file_report)
    rp.plot_summary_metrics(model_name, output_path, file_report)

    result = {
        'Modelo': key,
        'Acurácia': round(max(history_f2.history['val_accuracy']), 4),
        'FPS (RTX 4060)': round(fps, 2),
        'VRAM_Estimada': "Alta" if "convnext" in model_name else "Baixa",
        "Total_Params": f"{model.count_params():,}",
        'Incerteza_Media': float(round(avg_incerteza, 4)),
        'Tempo_FT_Seg': round(train_time, 2)
    }
    
    save_result(result)
    return result

if __name__ == "__main__":
    # modelos = ['resnet50']
    seeds = [42,45,47]
    start_process = time.time()
    modelos_executados = load_results() 
    if not os.path.exists(CONFIG['OUTPUT_MODEL']):
        os.makedirs(CONFIG['OUTPUT_MODEL'], exist_ok=True)
        print(f"Diretório criado: {CONFIG['OUTPUT_MODEL']}")

    folder_name_output = os.path.basename(CONFIG['OUTPUT_MODEL'])    
    modelos = list(MODEL_FACTORY.keys())
    data_aug = CONFIG['APPLY_DATA_AUGMENTATION'] # Ativa o Data Augmentation agressivo
    
    all_results = []
    for m in modelos:
        model_runs = []
        
        for seed in seeds:
            print(f"\n--- Modelo: {m} | Seed: {seed} ---")

            result = run_experiment(m, data_aug, seed)
            result['seed'] = seed

            model_runs.append(result)

        all_results.append({
            'modelo': m,
            'runs': model_runs
        })
    
    final_results = []

    for model_data in all_results:
        accs = [r['Acurácia'] for r in model_data['runs']]
        fpss = [r['FPS (RTX 4060)'] for r in model_data['runs']]

        final_results.append({
            'Modelo': model_data['modelo'],
            'Acc_Media': np.mean(accs),
            'Acc_Std': np.std(accs),
            'FPS_Medio': np.mean(fpss),
        })
    
    results = list(load_results().values())    
    rp.plot_comparison_results(results)
    rp.gerar_graficos_avancados(results)
    
    utils.save_results_csv(results, CONFIG['OUTPUT_MODEL'], 'benchmark_rtx4060.csv')
    print(f'Tempo total de processamento: {time.time() - start_process:.2f} segundos')