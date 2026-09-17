import os
import sys
import time
import cv2

# Ajusta o path para importar os módulos da pasta raiz de producao
BASE_PRODUCAO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PRODUCAO_DIR not in sys.path:
    sys.path.insert(0, BASE_PRODUCAO_DIR)

from config import Config
from train import aplicar_poligono_sprint2


def salvar_imagens_poligono():
    src_dir = Config.PROD_DATASET_DIR
    dst_dir = os.path.abspath(os.path.join(Config.BASE_DIR, "..", "data", "poligono"))
    os.makedirs(dst_dir, exist_ok=True)

    classes = [
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith('.')
    ]
    classes = sorted(classes)
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    total_salvas = 0

    print(f"[INFO] Processando {len(classes)} classes de '{src_dir}' para '{dst_dir}' (Apenas ROI Poligonal)...")
    start_time = time.time()

    for cls in classes:
        cls_src = os.path.join(src_dir, cls)
        cls_dst = os.path.join(dst_dir, cls)
        os.makedirs(cls_dst, exist_ok=True)

        arquivos = [f for f in os.listdir(cls_src) if f.lower().endswith(extensions)]
        print(f"[INFO] Classe '{cls}': {len(arquivos)} imagens...")

        for f in arquivos:
            src_file = os.path.join(cls_src, f)
            dst_file = os.path.join(cls_dst, f)

            img = cv2.imread(src_file)
            if img is None:
                continue

            # Carrega a imagem, aplica o ROI e salva
            img_roi = aplicar_poligono_sprint2(img)
            cv2.imwrite(dst_file, img_roi)
            total_salvas += 1

    elapsed = time.time() - start_time
    print(f"[INFO] Sucesso! Total de {total_salvas} imagens salvas em '{dst_dir}' em {elapsed:.2f}s.")


if __name__ == "__main__":
    salvar_imagens_poligono()
