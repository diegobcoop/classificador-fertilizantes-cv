import os

# Obtém o caminho raiz do projeto de forma dinâmica e portável
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", "..", ".."))

CONFIG = {
    'IMAGE_SIZE': (224, 224),
    'BATCH_SIZE': 32,
    'INITIAL_EPOCHS': 30,    # Épocas para a Fase 1
    'FINE_TUNE_EPOCHS': 50,  # Épocas para a Fase 2 (Ajuste Fino)
    'INITIAL_LR': 1e-3,      # Taxa de aprendizado inicial = 0.001
    'FINE_TUNE_LR': 1e-5,    # Taxa muito baixa para o Fine-Tuning = 0.00001
    'UNFREEZE_LAYERS': 20,   # Descongela apenas as últimas 20 camadas do modelo base
    'PATIENCE': 30,          # Épocas para early stopping
    'DROPOUT': 0.3,          # Dropout para regularização
    'TRAIN_DIR': os.path.join(_PROJECT_ROOT, 'data', 'processadas_limpas'),
    'VAL_DIR': os.path.join(_PROJECT_ROOT, 'data', 'processadas_limpas'),
    'OUTPUT_MODEL': os.path.join(_PROJECT_ROOT, 'output', 'tensorflow'),
    'APPLY_DATA_AUGMENTATION': True, # Habilita ou desabilita o uso de data augmentation
}