import os
import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import classification_report
from core.config import CONFIG
from core import reporting as rp
from core import utils

def run_validation_pipeline():
    # Configurações de caminhos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolvendo caminhos baseados na raiz do projeto
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    model_path = os.path.abspath(os.path.join(project_root, "output", "yolo", "PROD_yolo26x-cls", "train", "weights", "best.pt"))
    val_dir = os.path.abspath(os.path.join(project_root, "data", "processadas_limpas", "validacao"))
    output_path = os.path.abspath(os.path.join(project_root, "output", "yolo", "PROD_yolo26x-cls", "validacao_pipeline"))

    os.makedirs(output_path, exist_ok=True)

    print(f"📦 Carregando o modelo YOLO de: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")
    model = YOLO(model_path)

    print(f"📂 Lendo imagens de validação de: {val_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Diretório de validação não encontrado: {val_dir}")
        
    # Obter os nomes das classes que constam no diretório de validação
    class_names = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    print(f"Classes identificadas no diretório de validação: {class_names}")

    # Criar mapeamento de classe para índice baseado no modelo
    # Isso garante o alinhamento com a ordem que o modelo aprendeu
    name_to_idx = {name: idx for idx, name in model.names.items()}
    print(f"Mapeamento de classes do modelo: {name_to_idx}")

    # Armazenar informações das predições
    y_true = []
    y_pred = []
    detalhes = []

    # Varre as pastas de validação
    for class_name in class_names:
        if class_name not in name_to_idx:
            print(f"⚠️ Aviso: Classe '{class_name}' não encontrada nas classes do modelo. Pulando...")
            continue
        
        class_idx = name_to_idx[class_name]
        class_path = os.path.join(val_dir, class_name)
        
        # Listar as imagens da pasta
        extensoes = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        img_names = [f for f in os.listdir(class_path) if f.lower().endswith(extensoes)]
        
        print(f"Processando classe '{class_name}' ({len(img_names)} imagens)...")
        
        for img_name in img_names:
            img_path = os.path.join(class_path, img_name)
            
            # 1. Carregar imagem
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Falha ao carregar imagem: {img_path}")
                continue
                
            # 2. Redimensionar para utils.IMAGE_SIZE (que é 224, 224)
            img_resized = cv2.resize(img, utils.IMAGE_SIZE)
            
            # 3. Executar predição no modelo
            start_time = time.time()
            results = model(img_resized, verbose=False)
            inference_time = time.time() - start_time
            
            # Extrair resultados
            probs = results[0].probs
            pred_idx = probs.top1
            pred_name = model.names[pred_idx]
            conf = float(probs.top1conf)
            
            y_true.append(class_idx)
            y_pred.append(pred_idx)
            
            detalhes.append({
                'imagem': img_name,
                'caminho': img_path,
                'classe_real': class_name,
                'classe_real_idx': class_idx,
                'classe_predita': pred_name,
                'classe_predita_idx': pred_idx,
                'confiança': conf,
                'tempo_infe_ms': inference_time * 1000
            })

    # Criar DataFrame com as predições individuais
    df_detalhes = pd.DataFrame(detalhes)
    detalhes_path = os.path.join(output_path, 'predicoes_detalhadas.csv')
    df_detalhes.to_csv(detalhes_path, index=False)
    print(f"💾 Predições detalhadas salvas em: {detalhes_path}")

    # Gerar classificação das métricas
    model_name = "PT_yolo26x-cls_validacao"
    present_classes = sorted(list(set(y_true)))
    target_names = [model.names[i] for i in present_classes]

    report = classification_report(
        y_true, 
        y_pred, 
        labels=present_classes,
        target_names=target_names, 
        output_dict=True
    )

    file_report = 'relatorio_classificacao_validacao.csv'
    utils.save_classification_report(report, file_report, output_path)

    # Mapear os IDs numéricos reais e previstos para as strings de nomes de classes para a matriz de confusão
    rp.plot_confusion_matrix(model_name, y_true, y_pred, target_names, output_path)
    rp.plot_report_metrics("yolo26x-cls_validacao", output_path, file_report)
    rp.plot_summary_metrics("yolo26x-cls_validacao", output_path, file_report)

    print("✅ Pipeline de validação concluído com sucesso!")

if __name__ == "__main__":
    run_validation_pipeline()
