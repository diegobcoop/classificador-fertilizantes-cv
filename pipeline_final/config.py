import os

class Config:
    # --- CAMINHOS BASE ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Caminho do novo dataset consolidado de produção (unificado para o YOLO)
    PROD_DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "originais"))
    
    # Caminho absoluto do modelo executável em produção
    MODELO_DIR = os.path.join(BASE_DIR, "modelo")
    MODEL_PATH = os.path.join(MODELO_DIR, "best.pt")
    
    # Caminhos relacionados à Curadoria de Imagens
    CURADORIA_DIR = os.path.join(BASE_DIR, "api", "curadoria")
    CURADORIA_IMGS_DIR = os.path.join(CURADORIA_DIR, "imagens")
    CURADORIA_DB_PATH = os.path.join(CURADORIA_DIR, "dados.json")
    
    # --- PARÂMETROS DO MODELO E TREINAMENTO ---
    YOLO_VARIANT = "yolo26s-cls"
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 100
    PATIENCE = 40
    LEARNING_RATE = 0.001
    DROPOUT = 0.2 # Desativa neurônios aleatoriamente para evitar dependência de ruído
    WEIGHT_DECAY=0.01 # Penaliza pesos muito grandes
    LABEL_SMOOTHING=0.1 # Suaviza a certeza excessiva entre classes contíguas

    
    # Lista padrão de classes de fertilizantes
    CLASSES = ['0to10', '11to20', '21to30', '31to40', '41to50', '51to60', '61to70', '71to80', '81to90', '91to100']
