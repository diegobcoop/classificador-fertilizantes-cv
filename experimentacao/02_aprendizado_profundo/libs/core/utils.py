import os
import cv2
import numpy as np
import pandas as pd

# ==============================
# CONFIG
# ==============================

INPUT_DIR = "//home/diegobaierle/Dev/pos/sprint3/data/originais"
OUTPUT_DIR = "/home/diegobaierle/Dev/pos/sprint3/data/processadas"
LORA_PATH = "/home/diegobaierle/Dev/pos/sprint3/data/loRa/lora_output"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")

IMAGE_SIZE = (224,224)

# ==============================
# FFT FILTER
# ==============================
def apply_fft_filter_rgb(image, radius=60):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channels = cv2.split(img)
    filtered_channels = []

    for ch in channels:
        f = np.fft.fft2(ch)
        fshift = np.fft.fftshift(f)

        rows, cols = ch.shape
        crow, ccol = rows//2, cols//2

        mask = np.zeros((rows,cols), np.uint8)
        cv2.circle(mask,(ccol,crow),radius,1,-1)

        fshift = fshift * mask

        ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(ishift)

        img_back = np.abs(img_back)
        img_back = np.clip(img_back,0,255)

        filtered_channels.append(img_back.astype(np.uint8))

    return cv2.merge(filtered_channels)

def count_images(train_dir):
    # Criar uma lista vazia para armazenar os nomes das pastas
    pastas = []
    extensoes = ('.jpg', '.jpeg', '.png', '.bmp')
    lista = sorted(os.listdir(train_dir))
    for d in lista:
        # Criamos o caminho completo para verificar se é diretório
        caminho_completo = os.path.join(train_dir, d)
        
        # Se for uma pasta, adicionamos à nossa lista
        if os.path.isdir(caminho_completo):
            pastas.append(d)

    # 2. Criar um dicionário vazio para as contagens
    contagens = {}
    for pasta in pastas:
        # Montamos o caminho da subpasta
        caminho_subpasta = os.path.join(train_dir, pasta)
        
        # Listamos os arquivos dentro dela
        arquivos_na_pasta = [f for f in os.listdir(caminho_subpasta) if f.lower().endswith(extensoes)]
        
        # Guardamos o nome da pasta como CHAVE e a quantidade como VALOR
        contagens[pasta] = len(arquivos_na_pasta)

    max_imagens = max(contagens.values())

    for classe, quantidade in contagens.items():
        print(f'Classe: {classe}, Quantidade: {quantidade}')
    
    total = sum(contagens.values())
    print(f'Total de imagens: {total}')
    return contagens, max_imagens


def save_training_history(history_f1, history_f2, model_name, output_path):
    # Converte os dicionários de histórico em DataFrames
    df1 = pd.DataFrame(history_f1.history)
    df2 = pd.DataFrame(history_f2.history)
    
    # Adiciona uma coluna para identificar a fase
    df1['fase'] = 'feature_extraction'
    df2['fase'] = 'fine_tuning'
    
    # Une os dois
    df_final = pd.concat([df1, df2], ignore_index=True)
    
    # Salva no seu diretório de output
    csv_path = os.path.join(output_path, f"history_{model_name}.csv")
    df_final.to_csv(csv_path, index=False)
    print(f"📈 Histórico de treino salvo em: {csv_path}")
    return df_final

def save_classification_report(report, file_name, output_path):
    df_report = pd.DataFrame(report).transpose()
    csv_path = os.path.join(output_path, file_name)
    df_report.to_csv(csv_path)
    print(f"📊 Relatório de classificação salvo em: {csv_path}")
    

def save_results_csv(results, output_path, file_name='benchmark_comparativo.csv'):
    df = pd.DataFrame(results).transpose()
    csv_path = os.path.join(output_path, file_name)
    df.to_csv(csv_path, index=False)
    print(f"📊 Resultados comparativos salvos em: {csv_path}")
    
    
def remove_dirty_images(dirty_dir, target_dirs, dry_run=True):
    total = 0

    for class_name in os.listdir(dirty_dir):
        dirty_class_path = os.path.join(dirty_dir, class_name)

        if not os.path.isdir(dirty_class_path):
            continue

        dirty_images = set(os.listdir(dirty_class_path))

        for target_base in target_dirs:
            target_class_path = os.path.join(target_base, class_name)

            if not os.path.exists(target_class_path):
                continue

            for img_name in dirty_images:
                target_img_path = os.path.join(target_class_path, img_name)

                if os.path.exists(target_img_path):
                    total += 1
                    if dry_run:
                        print(f"[SIMULAÇÃO] {target_img_path}")
                    else:
                        os.remove(target_img_path)
                        print(f"[REMOVIDO] {target_img_path}")

    print(f"\nTotal encontrado: {total}")