import os
import time
import torch
import shutil
from ultralytics import YOLO
from core.config import CONFIG

# --- CONFIGURAÇÃO DE AMBIENTE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Dispositivo de execução: {device}")

def merge_splits_to_production(src_dir, dst_dir):
    """
    Combina todas as imagens de 'train', 'val' e 'validacao' em um único
    diretório unificado de produção (com subpastas train e val idênticas para o YOLO).
    Isso garante que o modelo treine com 100% dos dados.
    """
    print(f"\n📂 Consolidando base de dados para produção...")
    print(f"  Origem: {os.path.abspath(src_dir)}")
    print(f"  Destino: {os.path.abspath(dst_dir)}")

    # Identificar as pastas de origem existentes
    splits = ['train', 'val']
    
    # Descobrir todas as classes presentes nas pastas
    classes = set()
    for split in splits:
        split_path = os.path.join(src_dir, split)
        if os.path.exists(split_path):
            for d in os.listdir(split_path):
                if os.path.isdir(os.path.join(split_path, d)):
                    classes.add(d)
    
    classes = sorted(list(classes))
    print(f"🏷️ Classes detectadas: {classes}")

    # Criar estruturas de pastas de treino e validação no destino
    # Para o YOLO classificador não falhar na validação, colocamos as mesmas imagens em ambas
    for split in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

    # Copiar ou linkar os arquivos de imagem
    extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    for cls in classes:
        all_images = []
        for split in splits:
            split_class_dir = os.path.join(src_dir, split, cls)
            if os.path.exists(split_class_dir):
                for f in os.listdir(split_class_dir):
                    if f.lower().endswith(extensions):
                        all_images.append(os.path.join(split_class_dir, f))
        
        print(f"  Class '{cls}': consolidando {len(all_images)} imagens...")
        
        for img_path in all_images:
            filename = os.path.basename(img_path)
            
            # Criamos os arquivos no split de 'train' e 'val'
            for split in ['train', 'val']:
                dst_path = os.path.join(dst_dir, split, cls, filename)
                
                # Evita recriar se o arquivo já existir
                if os.path.exists(dst_path):
                    continue
                
                try:
                    # Tenta criar Hard Link para economizar espaço em disco e ser instantâneo
                    os.link(img_path, dst_path)
                except Exception:
                    # Fallback para cópia física tradicional se estiver cruzando volumes de disco
                    shutil.copy2(img_path, dst_path)

    print("✅ Consolidação do dataset finalizada com sucesso!")


def run_production_training(variant="yolo26x-cls"):
    """
    Executa o treinamento final (produção) utilizando o dataset consolidado.
    """
    model_name = f"PROD_{variant}"
    print(f"\n🚀 INICIANDO TREINAMENTO DE PRODUÇÃO: {model_name}")

    # Diretórios do Script e do Dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Caminho absoluto da pasta original de dados
    src_dataset_dir = os.path.abspath(CONFIG['TRAIN_DIR'])
    
    # Caminho do novo dataset consolidado de produção (processadas_limpas_production)
    dst_dataset_name = os.path.basename(os.path.normpath(src_dataset_dir)) + "_production"
    dst_dataset_dir = os.path.join(os.path.dirname(src_dataset_dir), dst_dataset_name)

    # 1. Unificar bases em um diretório temporário/definitivo de produção
    merge_splits_to_production(src_dataset_dir, dst_dataset_dir)

    # Caminho de output da produção em output/yolo
    output_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "output", "yolo"))
    output_path = os.path.join(output_base, model_name)
    os.makedirs(output_path, exist_ok=True)

    # 2. Carregar o modelo YOLO base pré-treinado
    print(f"📦 Carregando modelo base: {variant}.pt")
    model = YOLO(f"{variant}.pt")

    # 3. Executar o Treinamento no dataset completo
    start_train = time.time()
    results_train = model.train(
        data=dst_dataset_dir,          # Dataset com todas as imagens (train + val + validacao)
        epochs=CONFIG['FINE_TUNE_EPOCHS'],
        imgsz=CONFIG['IMAGE_SIZE'][0],
        project=output_path,
        exist_ok=True,
        patience=CONFIG['PATIENCE'],
        lr0=0.001,                     # Hiperparâmetros recomendados para o ajuste fino
        lrf=0.01,
        degrees=10.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.0,
        mixup=0.0,
        batch=CONFIG['BATCH_SIZE'],
        device=0
    )
    train_time = time.time() - start_train
    print(f"🏁 Treinamento concluído em {train_time/60:.2f} minutos.")
    print(f"💾 Modelo final de produção salvo em: {os.path.join(output_path, 'train', 'weights', 'best.pt')}")


if __name__ == "__main__":
    # Pode alterar para o modelo YOLO de sua preferência (ex: yolo11n-cls, yolo26x-cls, etc.)
    modelo_escolhido = "yolo26x-cls"
    run_production_training(modelo_escolhido)
