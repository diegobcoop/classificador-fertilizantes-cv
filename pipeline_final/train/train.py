from PIL import ImageCms
import os
import sys
import time
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ajusta o path para importar os módulos da pasta raiz de producao
BASE_PRODUCAO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PRODUCAO_DIR not in sys.path:
    sys.path.insert(0, BASE_PRODUCAO_DIR)

from config import Config
from image_processing import (
    redimensionar_com_letterbox,
    converter_para_escala_cinza,
    aplicar_clahe
)


# Polígono de interesse (ROI) validado na Sprint 2 (referência 1280x720)
PONTOS_ROI_SPRINT2 = np.array(
    [[10, 700], [10, 400], [500, 50], [800, 50], [1270, 400], [1270, 700]],
    dtype=np.int32
)


def aplicar_poligono_sprint2(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica a máscara poligonal da ROI da Sprint 2 (método aplicarPoligono do utils.py).
    Garante a resolução de referência (1280x720) para alinhamento dos vértices.
    """
    if imagem.shape[1] != 1280 or imagem.shape[0] != 720:
        imagem = cv2.resize(imagem, (1280, 720))
        
    mascara = np.zeros(imagem.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [PONTOS_ROI_SPRINT2], 255)
    return cv2.bitwise_and(imagem, imagem, mask=mascara)


def preprocessar_imagem_treino(imagem: np.ndarray) -> np.ndarray:
    """
    Aplica o pipeline padronizado de pré-processamento para imagens de treino:
    1. Aplicação do ROI poligonal da Sprint 2 (antes de redimensionar).
    2. Redimensionamento usando letterbox para (640, 640).
    3. Tratamento das cores: conversão para escala de cinza mantendo 3 canais e aplicação de CLAHE.
    """
    # 1. Aplica ROI da Sprint 2 antes do redimensionamento
    imagem = aplicar_poligono_sprint2(imagem)
    
    # 2. Redimensiona com letterbox para (640, 640)
    imagem = redimensionar_com_letterbox(imagem, tamanho_destino=(640, 640))
        
    # 3. Tratamento de cores e contraste (Grayscale 3 canais + CLAHE)
    imagem = aplicar_clahe(
        converter_para_escala_cinza(imagem, manter_3_canais=True),
        limite_contraste=2.0,
        tamanho_grade=(8, 8)
    )
    
    return imagem


def preparar_dataset_treinamento(src_dir: str, dst_dir: str):
    """
    Prepara o dataset definitivo aplicando o pipeline em 100% das imagens.
    
    Como o YOLO Classification exige tecnicamente a presença de 'train' e 'val'
    na estrutura de pastas do dataset:
    - 100% das imagens são colocadas na pasta 'train' (usadas para otimizar os pesos).
    - As mesmas imagens são espelhadas em 'val' (apenas para satisfazer o requisito do YOLO).
    Dessa forma, NENHUMA imagem é reservada ou descartada do treinamento.
    """
    print("[INFO] Preparando dataset definitivo de produção (100% dos dados para treino)...")
    print(f"       Diretório Fonte: {src_dir}")
    print(f"       Diretório de Trabalho: {dst_dir}")

    classes = [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith('.')
    ]
    classes = sorted(classes)
    print(f"[INFO] Classes encontradas ({len(classes)}): {classes}")

    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    total_processadas = 0

    for cls in classes:
        cls_src_dir = os.path.join(src_dir, cls)
        cls_train_dst = os.path.join(dst_dir, "train", cls)
        cls_val_dst = os.path.join(dst_dir, "val", cls)
        
        os.makedirs(cls_train_dst, exist_ok=True)
        os.makedirs(cls_val_dst, exist_ok=True)
        
        arquivos = [f for f in os.listdir(cls_src_dir) if f.lower().endswith(extensions)]
        print(f"[INFO] Processando classe '{cls}' ({len(arquivos)} imagens)...")
        
        for filename in arquivos:
            src_file = os.path.join(cls_src_dir, filename)
            dst_train_file = os.path.join(cls_train_dst, filename)
            dst_val_file = os.path.join(cls_val_dst, filename)
            
            # Lê a imagem
            img = cv2.imread(src_file)
            if img is None:
                print(f"[WARNING] Erro ao ler imagem '{src_file}', ignorando.")
                continue
            
            # Aplica o pipeline
            img_proc = preprocessar_imagem_treino(img)
            
            # Salva na pasta train (100% das imagens treinam o modelo)
            cv2.imwrite(dst_train_file, img_proc)
            
            # Cria link/cópia para val apenas para cumprir a exigência de formato do YOLO
            if not os.path.exists(dst_val_file):
                try:
                    os.link(dst_train_file, dst_val_file)
                except Exception:
                    shutil.copy2(dst_train_file, dst_val_file)
            
            total_processadas += 1

    print(f"[INFO] Preparação concluída. Total de {total_processadas} imagens prontas para treino total.")


def treinar_modelo():
    """
    Executa o treinamento final e definitivo utilizando 100% das imagens da pasta data/treinamento.
    """
    print(f"[INFO] Iniciando treinamento definitivo com variante: {Config.YOLO_VARIANT}")
    
    # 1. Diretório de trabalho temporário do dataset pré-processado para o YOLO
    dataset_treino_yolo = os.path.join(Config.BASE_DIR, "train", "dataset_yolo")
    
    # Prepara todas as imagens da pasta data/treinamento
    preparar_dataset_treinamento(Config.PROD_DATASET_DIR, dataset_treino_yolo)

    # 2. Carregar modelo YOLO inicial pré-treinado
    print(f"[INFO] Carregando modelo base: {Config.YOLO_VARIANT}.pt")
    model = YOLO(f"{Config.YOLO_VARIANT}.pt")

    # 3. Executar o Treinamento Definitivo
    runs_dir = os.path.abspath(os.path.join(Config.BASE_DIR, "..", "output", "pipeline_final", "runs"))
    os.makedirs(runs_dir, exist_ok=True)
    
    start_time = time.time()
    results = model.train(
        data=dataset_treino_yolo,
        epochs=Config.EPOCHS,
        imgsz=Config.IMAGE_SIZE[0],
        project=runs_dir,
        exist_ok=True,
        patience=Config.PATIENCE,
        lr0=Config.LEARNING_RATE,
        lrf=0.01,
        degrees=25.0,     # ~25 graus (Data Augmentation equivalente a RandomRotation(0.14) do validacao_tf.py)
        translate=0.2,    # 20% de translação (equivalente a RandomTranslation(0.2, 0.2))
        scale=0.2,        # 20% de ganho/zoom (equivalente a RandomZoom(0.2))
        fliplr=0.5,       # 50% de probabilidade de flip horizontal (equivalente a RandomFlip("horizontal"))
        flipud=0.0,       # Sem flip vertical
        mosaic=0.0,
        mixup=0.0,
        batch=Config.BATCH_SIZE,
        val=False, # Não gasta tempo separando validação; treina com 100% dos dados
        device=0 if torch.cuda.is_available() else "cpu",
        dropout=Config.DROPOUT,       # Desativa neurônios aleatoriamente para evitar dependência de ruído
        weight_decay=Config.WEIGHT_DECAY, # Penaliza pesos muito grandes (regularização L2)
        label_smoothing=Config.LABEL_SMOOTHING # Suaviza a certeza excessiva entre classes contíguas
    )
    
    elapsed_time = time.time() - start_time
    print(f"[INFO] Treinamento concluído em {elapsed_time/60:.2f} minutos.")
    
    # 4. Copiar o peso final gerado para o modelo executável de produção
    # No treinamento completo, 'last.pt' representa o aprendizado de todas as épocas com todas as imagens
    weights_dir = os.path.join(runs_dir, "train", "weights")
    last_weight_path = os.path.join(weights_dir, "last.pt")
    best_weight_path = os.path.join(weights_dir, "best.pt")
    
    final_weight = best_weight_path if os.path.exists(best_weight_path) else last_weight_path
    
    if os.path.exists(final_weight):
        os.makedirs(Config.MODELO_DIR, exist_ok=True)
        shutil.copy2(final_weight, Config.MODEL_PATH)
        print(f"[INFO] Peso de produção atualizado com sucesso em: {Config.MODEL_PATH}")
    else:
        print(f"[WARNING] Nenhum arquivo de peso encontrado em: {weights_dir}")


if __name__ == "__main__":
    treinar_modelo()
