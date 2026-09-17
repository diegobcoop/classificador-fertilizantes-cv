import os
import time
import json
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 

from core.config import CONFIG
from core import reporting as rp
from core import utils
from sklearn.metrics import classification_report

# Bibliotecas PyTorch e YOLO
from ultralytics import YOLO
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- CONFIGURAÇÃO DE AMBIENTE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def get_output_yolo_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "yolo"))

def load_results():
    path = os.path.join(get_output_yolo_dir(), 'benchmark_rtx4060.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_result(result):
    results = load_results()
    results[result['Modelo']] = result
    path = os.path.join(get_output_yolo_dir(), 'benchmark_rtx4060.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# --- MODELO 1: YOLO CLASSIFICATION ---
def run_yolo_cls_experiment(variant):
    model_name = f"PT_{variant}"
    print(f"\n🚀 PROCESSANDO: {model_name}")
    
    output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "yolo"))
    output_path = os.path.join(output_base, model_name)
    os.makedirs(output_path, exist_ok=True)
    
    # O YOLO-cls aceita o caminho da pasta raiz que contém train/test
    model = YOLO(f"{variant}.pt")
    
    start_train = time.time()
    # Treino automático usando sua estrutura de pastas
    results_train = model.train(
        data=CONFIG['TRAIN_DIR'], # Pasta que contém 'train' e 'val'
        epochs=CONFIG['FINE_TUNE_EPOCHS'],
        imgsz=CONFIG['IMAGE_SIZE'][0],
        project=output_path,
        exist_ok=True,
        patience=CONFIG['PATIENCE'],
        lr0=0.001,            # LR muito mais conservador para AdamW (1e-3)
        lrf=0.01,
        # Augmentation: Foque no que não destrói a percepção de volume
        degrees=10.0,         # Rotação leve
        flipud=0.0,           # Desative se a câmera for fixa (evita confusão de "fundo" vs "topo")
        fliplr=0.5,           # Mirror horizontal é seguro
        mosaic=0.0,           # Desative para classificação de nível
        mixup=0.0,            # Desative para classes ordinais de volume
        batch=CONFIG['BATCH_SIZE'],
        device=0
    )
    train_time = time.time() - start_train

    # Validação e Métricas
    metrics = model.val(
        project=output_path
    )
    acc = metrics.top1
    
    res = {
        'Modelo': model_name,
        'Acurácia': round(float(acc), 4),
        'FPS (RTX 4060)': round(1000 / metrics.speed['inference'], 2),
        'Total_Params': "0",
        'Tempo_FT_Seg': round(train_time, 2)
    }
    save_result(res)
    return res


if __name__ == "__main__":
    start_process = time.time()
    modelos_executados = load_results()
    
    # 1. Executar YOLO Classification (Usa suas pastas atuais)
    yolo_models = ['yolo11n-cls',  
                   'yolo11s-cls', 
                   'yolo11m-cls', 
                   'yolo11l-cls', 
                   'yolo11x-cls', 
                   'yolo26n-cls', 
                   'yolo26s-cls', 
                   'yolo26m-cls', 
                   'yolo26l-cls', 
                   'yolo26x-cls']
    
    for m in yolo_models:
        if f"PT_{m}" not in modelos_executados:
            run_yolo_cls_experiment(m)
        else:
            print(f"⏩ {m} já processado.")

    # 2. Gerar Gráficos Finais (Mesma lógica do TF)
    results_list = list(load_results().values())
    
    # rp.plot_comparison_results(results_list)
    utils.save_results_csv(results_list, get_output_yolo_dir(), 'benchmark_rtx4060.csv')
    
    print(f'✅ Processamento PyTorch concluído em: {time.time() - start_process:.2f}s')