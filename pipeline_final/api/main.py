import os
import io
import time
import json
import uuid
import shutil
import cv2
import numpy as np
import base64
from datetime import datetime
from threading import Lock
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from ultralytics import YOLO

# Ajusta o path para importar o Config e image_processing da raiz de producao
import sys
BASE_PRODUCAO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_PRODUCAO_DIR not in sys.path:
    sys.path.append(BASE_PRODUCAO_DIR)

from config import Config
from image_processing import (
    processar_imagem_producao,
    CONFIGURACAO_PADRAO,
    converter_para_escala_cinza,
    aplicar_clahe,
    redimensionar_com_letterbox
)

# Garantir a criação das pastas necessárias no startup
os.makedirs(Config.CURADORIA_IMGS_DIR, exist_ok=True)
if not os.path.exists(Config.CURADORIA_DB_PATH):
    with open(Config.CURADORIA_DB_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=4)

# Lock para garantir operações thread-safe no banco JSON
db_lock = Lock()

# Inicializa o FastAPI
app = FastAPI(
    title="Fertilizantes YOLO Classifier API & Curadoria",
    description="API de producao para classificacao de fertilizantes com pipeline completo e curadoria ativa",
    version="1.2.0"
)

# Variáveis globais para o modelo
model = None
device = "cpu"

@app.on_event("startup")
def load_model():
    global model, device
    print(f"[INFO] Carregando modelo YOLO de: {Config.MODEL_PATH}")
    if not os.path.exists(Config.MODEL_PATH):
        print(f"[ERROR] Modelo nao encontrado em: {Config.MODEL_PATH}")
        return
    
    try:
        model = YOLO(Config.MODEL_PATH)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Modelo YOLO carregado com sucesso no dispositivo: {device}")
    except Exception as e:
        print(f"[ERROR] Erro ao carregar o modelo YOLO: {e}")


# --- FUNÇÕES DE BANCO DE DADOS (JSON) ---
def read_db():
    with db_lock:
        if not os.path.exists(Config.CURADORIA_DB_PATH):
            return []
        try:
            with open(Config.CURADORIA_DB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

def write_db(data):
    with db_lock:
        with open(Config.CURADORIA_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# --- CLASSES PYDANTIC ---
class DecisionBody(BaseModel):
    id: str
    action: str  # "confirm" ou "correct"
    final_class: str


# --- FUNÇÃO AUXILIAR DE PREDIÇÃO YOLO ---
def predizer_imagem_cv2(img_cv2: np.ndarray):
    """
    Executa predição em um array BGR do OpenCV e retorna métricas estruturadas.
    """
    start_time = time.time()
    results = model(img_cv2, verbose=False)
    inference_time_ms = (time.time() - start_time) * 1000
    
    probs = results[0].probs
    pred_idx = int(probs.top1)
    pred_name = model.names[pred_idx]
    conf = float(probs.top1conf)
    
    all_confidences = {}
    for idx, name in model.names.items():
        all_confidences[name] = float(probs.data[idx])
        
    all_confidences_sorted = dict(
        sorted(all_confidences.items(), key=lambda item: item[1], reverse=True)
    )
    
    return {
        "predicted_class": pred_name,
        "class_index": pred_idx,
        "confidence": conf,
        "inference_time_ms": round(inference_time_ms, 2),
        "all_confidences": all_confidences_sorted
    }


def cv2_to_base64(img_bgr: np.ndarray) -> str:
    """
    Converte imagem OpenCV para string base64 PNG para exibição no frontend.
    """
    success, buffer = cv2.imencode('.png', img_bgr)
    if not success:
        return ""
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


# --- ENDPOINTS DA API ---

@app.get("/health")
def health_check():
    """
    Status de integridade da API.
    """
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_name": Config.YOLO_VARIANT,
        "device": device,
        "model_path": Config.MODEL_PATH,
        "timestamp": time.time()
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("camera_split")  # 'camera_split' (padrão produção) ou 'single_crop'
):
    """
    Classifica a imagem com base no modo selecionado:
    - 'camera_split': Aplica pipeline de produção da Câmera Fixa (Split X=1050, Espelhamento do Direito, Rotações, Máscaras ROI, Letterbox, Grayscale e CLAHE).
    - 'single_crop': Trata a imagem como um recorte individual (apenas Letterbox se > 640, Grayscale e CLAHE).
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo YOLO indisponivel.")
    
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400, 
            detail=f"Formato de arquivo invalido. Suportados: {', '.join(valid_extensions)}"
        )
    
    try:
        contents = await file.read()
        
        # Decodificar imagem original
        nparr = np.frombuffer(contents, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_original is None:
            raise HTTPException(status_code=400, detail="Nao foi possivel decodificar a imagem.")
        
        file_ext = os.path.splitext(file.filename)[1]
        timestamp_now = datetime.now().isoformat()
        db_entries = []
        
        # MODO 1: CÂMERA FIXA (2 BOXES: ESQUERDO E DIREITO)
        if mode == "camera_split":
            # Aplica o pipeline oficial
            (box_esq_proc, box_dir_proc), _ = processar_imagem_producao(
                img_original,
                config=CONFIGURACAO_PADRAO,
                retornar_etapas=False
            )
            
            # Predição Box Esquerdo
            pred_esq = predizer_imagem_cv2(box_esq_proc)
            id_esq = str(uuid.uuid4())
            path_esq = os.path.join(Config.CURADORIA_IMGS_DIR, f"{id_esq}_esq.jpg")
            cv2.imwrite(path_esq, box_esq_proc)
            
            # Predição Box Direito
            pred_dir = predizer_imagem_cv2(box_dir_proc)
            id_dir = str(uuid.uuid4())
            path_dir = os.path.join(Config.CURADORIA_IMGS_DIR, f"{id_dir}_dir.jpg")
            cv2.imwrite(path_dir, box_dir_proc)
            
            # Registra no banco de curadoria
            db_entries.append({
                "id": id_esq,
                "filename": f"{file.filename} (Box Esquerdo)",
                "temp_image_path": path_esq,
                "predicted_class": pred_esq["predicted_class"],
                "class_index": pred_esq["class_index"],
                "confidence": pred_esq["confidence"],
                "inference_time_ms": pred_esq["inference_time_ms"],
                "timestamp": timestamp_now,
                "status": "pending",
                "final_class": None
            })
            db_entries.append({
                "id": id_dir,
                "filename": f"{file.filename} (Box Direito)",
                "temp_image_path": path_dir,
                "predicted_class": pred_dir["predicted_class"],
                "class_index": pred_dir["class_index"],
                "confidence": pred_dir["confidence"],
                "inference_time_ms": pred_dir["inference_time_ms"],
                "timestamp": timestamp_now,
                "status": "pending",
                "final_class": None
            })
            
            db_data = read_db()
            db_data.extend(db_entries)
            write_db(db_data)
            
            return JSONResponse(content={
                "success": True,
                "mode": "camera_split",
                "filename": file.filename,
                "device": device,
                "boxes": {
                    "esquerdo": {
                        "id": id_esq,
                        "title": "Box Esquerdo",
                        "image_base64": cv2_to_base64(box_esq_proc),
                        "predicted_class": pred_esq["predicted_class"],
                        "class_index": pred_esq["class_index"],
                        "confidence": pred_esq["confidence"],
                        "inference_time_ms": pred_esq["inference_time_ms"],
                        "all_confidences": pred_esq["all_confidences"]
                    },
                    "direito": {
                        "id": id_dir,
                        "title": "Box Direito",
                        "image_base64": cv2_to_base64(box_dir_proc),
                        "predicted_class": pred_dir["predicted_class"],
                        "class_index": pred_dir["class_index"],
                        "confidence": pred_dir["confidence"],
                        "inference_time_ms": pred_dir["inference_time_ms"],
                        "all_confidences": pred_dir["all_confidences"]
                    }
                }
            })
            
        # MODO 2: RECORTE INDIVIDUAL / HISTÓRICA
        else:
            h, w = img_original.shape[:2]
            img_proc = img_original.copy()
            if w > 640 or h > 640:
                img_proc = redimensionar_com_letterbox(img_proc, tamanho_destino=(640, 640))
                
            img_proc = aplicar_clahe(
                converter_para_escala_cinza(img_proc, manter_3_canais=True),
                limite_contraste=2.0,
                tamanho_grade=(8, 8)
            )
            
            pred = predizer_imagem_cv2(img_proc)
            unique_id = str(uuid.uuid4())
            temp_path = os.path.join(Config.CURADORIA_IMGS_DIR, f"{unique_id}.jpg")
            cv2.imwrite(temp_path, img_proc)
            
            db_entries.append({
                "id": unique_id,
                "filename": file.filename,
                "temp_image_path": temp_path,
                "predicted_class": pred["predicted_class"],
                "class_index": pred["class_index"],
                "confidence": pred["confidence"],
                "inference_time_ms": pred["inference_time_ms"],
                "timestamp": timestamp_now,
                "status": "pending",
                "final_class": None
            })
            
            db_data = read_db()
            db_data.extend(db_entries)
            write_db(db_data)
            
            return JSONResponse(content={
                "success": True,
                "mode": "single_crop",
                "id": unique_id,
                "filename": file.filename,
                "image_base64": cv2_to_base64(img_proc),
                "predicted_class": pred["predicted_class"],
                "class_index": pred["class_index"],
                "confidence": pred["confidence"],
                "inference_time_ms": pred["inference_time_ms"],
                "device": device,
                "all_confidences": pred["all_confidences"]
            })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento: {str(e)}")


# --- ENDPOINTS DA CURADORIA ---

@app.get("/api/curadoria/pendentes")
def get_pendentes():
    """
    Lista registros pendentes.
    """
    data = read_db()
    pendentes = [item for item in data if item["status"] == "pending"]
    pendentes.reverse()
    return pendentes

@app.get("/api/curadoria/imagem/{item_id}")
def get_curadoria_image(item_id: str):
    """
    Serve a imagem da fila temporaria.
    """
    data = read_db()
    item = next((i for i in data if i["id"] == item_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado.")
    
    img_path = item["temp_image_path"]
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Arquivo nao encontrado.")
        
    return FileResponse(img_path)

@app.post("/api/curadoria/decidir")
def decide_curate(body: DecisionBody):
    """
    Move a imagem para a pasta final de treino com base na curadoria.
    """
    data = read_db()
    item = next((i for i in data if i["id"] == body.id), None)
    
    if not item:
        raise HTTPException(status_code=404, detail="Item nao encontrado.")
    
    if item["status"] != "pending":
        raise HTTPException(status_code=400, detail="Curadoria ja executada para este item.")
    
    classes_validas = list(model.names.values()) if model else Config.CLASSES
    if body.final_class not in classes_validas:
        raise HTTPException(status_code=400, detail=f"Classe invalida: {body.final_class}")
    
    # 1. Definir o caminho de destino no dataset de treinamento
    target_dir = os.path.abspath(
        os.path.join(Config.PROD_DATASET_DIR, body.final_class)
    )
    os.makedirs(target_dir, exist_ok=True)
    
    original_ext = os.path.splitext(item["filename"])[1]
    if not original_ext:
        original_ext = ".jpg"
    sanitized_name = "".join(c for c in item["filename"] if c.isalnum() or c in "._-")
    new_filename = f"curated_{item['id'][:8]}_{sanitized_name}"
    
    if not new_filename.endswith(original_ext):
        new_filename += original_ext
        
    target_path = os.path.join(target_dir, new_filename)
    
    # 2. Mover o arquivo fisicamente
    src_path = item["temp_image_path"]
    if not os.path.exists(src_path):
        raise HTTPException(status_code=404, detail="Arquivo original nao encontrado.")
        
    try:
        shutil.move(src_path, target_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao mover arquivo: {e}")
    
    # 3. Atualizar o banco
    item["status"] = "confirmed" if body.action == "confirm" else "corrected"
    item["final_class"] = body.final_class
    item["curated_at"] = datetime.now().isoformat()
    item["saved_dataset_path"] = target_path
    
    write_db(data)
    
    print(f"[INFO] Curadoria: Item {item['id'][:8]} finalizado. Classe: {body.final_class}")
    
    return {
        "success": True,
        "message": f"Imagem salva com sucesso na pasta de treino da classe '{body.final_class}'",
        "saved_filename": new_filename
    }


# --- TEMPLATES HTML ESTÁTICOS ---

SHARED_CSS = """
:root {
    --bg-gradient: radial-gradient(circle at 50% 50%, #121520 0%, #08090d 100%);
    --glass-bg: rgba(20, 24, 38, 0.65);
    --glass-border: rgba(255, 255, 255, 0.08);
    --glass-highlight: rgba(255, 255, 255, 0.03);
    --primary: #4f46e5;
    --primary-glow: rgba(79, 70, 229, 0.4);
    --accent: #06b6d4;
    --accent-glow: rgba(6, 182, 212, 0.3);
    --text-main: #f3f4f6;
    --text-muted: #9ca3af;
    --success: #10b981;
    --warning: #f59e0b;
    --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: var(--bg-gradient);
    color: var(--text-main);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Nav Bar Styles */
.nav-bar {
    width: 100%;
    max-width: 1100px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1rem 2.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    z-index: 10;
    box-shadow: var(--card-shadow);
}
.nav-logo {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    color: var(--text-main);
}
.nav-logo span {
    color: var(--accent);
}
.nav-links {
    display: flex;
    gap: 1.5rem;
}
.nav-link {
    color: var(--text-muted);
    text-decoration: none;
    font-size: 0.95rem;
    font-weight: 600;
    transition: all 0.3s;
    position: relative;
    padding: 0.25rem 0;
}
.nav-link:hover, .nav-link.active {
    color: var(--text-main);
}
.nav-link.active::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
    border-radius: 2px;
}

/* Base Panel Styles */
.panel {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 2.5rem;
    box-shadow: var(--card-shadow);
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}

.panel::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 100%;
    background: linear-gradient(135deg, var(--glass-highlight) 0%, transparent 100%);
    pointer-events: none;
}

/* Action Button */
.btn {
    background: linear-gradient(135deg, var(--primary) 0%, #312e81 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 1rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    box-shadow: 0 4px 15px var(--primary-glow);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
}

.btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
    background: linear-gradient(135deg, #6366f1 0%, var(--primary) 100%);
}

.btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

/* Empty State */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    text-align: center;
    gap: 1rem;
    padding: 2rem 0;
}

.empty-state svg {
    width: 64px;
    height: 64px;
    opacity: 0.3;
}

/* Decorative Background Glows */
.glow-bg {
    position: absolute;
    width: 450px;
    height: 450px;
    border-radius: 50%;
    filter: blur(120px);
    z-index: 1;
    pointer-events: none;
    opacity: 0.3;
}

.glow-1 {
    top: -100px;
    left: -100px;
    background: var(--primary);
}

.glow-2 {
    bottom: -150px;
    right: -100px;
    background: var(--accent);
}
"""

INDEX_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fertilizantes YOLO Classifier API</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        {SHARED_CSS}
        
        .container {
            width: 100%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 420px 1fr;
            gap: 2rem;
            z-index: 10;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .header {
            grid-column: 1 / -1;
            text-align: center;
            margin-bottom: 1rem;
        }

        .header h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 30%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 300;
        }

        .api-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(6, 182, 212, 0.1);
            border: 1px solid rgba(6, 182, 212, 0.2);
            color: var(--accent);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.75rem;
        }

        .badge-dot {
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
            box-shadow: 0 0 10px var(--success);
        }

        /* Pipeline Mode Selector Switch */
        .mode-selector-box {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .mode-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-main);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .mode-options {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }

        .mode-option {
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 0.6rem 0.5rem;
            text-align: center;
            cursor: pointer;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            transition: all 0.25s ease;
        }

        .mode-option:hover {
            border-color: rgba(6, 182, 212, 0.4);
            color: var(--text-main);
        }

        .mode-option.active {
            background: linear-gradient(135deg, rgba(79, 70, 229, 0.25) 0%, rgba(6, 182, 212, 0.25) 100%);
            border-color: var(--accent);
            color: #fff;
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.2);
        }

        /* Dropzone Styles */
        .dropzone-container {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            justify-content: center;
            height: 100%;
        }

        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: 18px;
            padding: 2.5rem 1.5rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            background: rgba(255, 255, 255, 0.01);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--accent);
            background: rgba(6, 182, 212, 0.03);
            box-shadow: 0 0 25px rgba(6, 182, 212, 0.1);
            transform: translateY(-2px);
        }

        .dropzone svg {
            width: 44px;
            height: 44px;
            fill: var(--text-muted);
            transition: fill 0.3s;
        }

        .dropzone:hover svg {
            fill: var(--accent);
        }

        .dropzone p {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        .dropzone span {
            color: var(--text-main);
            font-weight: 600;
        }

        #fileInput {
            display: none;
        }

        /* Preview Section */
        .preview-box {
            width: 100%;
            height: 180px;
            border-radius: 14px;
            border: 1px solid var(--glass-border);
            background-size: cover;
            background-position: center;
            display: none;
            position: relative;
            overflow: hidden;
        }

        .preview-box.active {
            display: block;
            animation: fadeIn 0.5s ease;
        }

        .preview-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.85));
            padding: 8px 12px;
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: #fff;
        }

        /* Results Layout */
        .results-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            height: 100%;
        }

        .results-grid-boxes {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 600px) {
            .results-grid-boxes {
                grid-template-columns: 1fr;
            }
        }

        .result-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.25rem;
            animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            display: flex;
            flex-direction: column;
        }

        .box-proc-preview {
            width: 100%;
            height: 150px;
            border-radius: 10px;
            border: 1px solid var(--glass-border);
            background: #000;
            object-fit: contain;
            margin-bottom: 1rem;
        }

        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .result-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-muted);
            font-weight: 700;
        }

        .inference-badge {
            font-size: 0.72rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 3px 8px;
            border-radius: 6px;
            color: var(--text-muted);
        }

        .predicted-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 800;
            color: #fff;
            margin-bottom: 0.25rem;
            background: linear-gradient(135deg, #fff 50%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Progress Bar */
        .progress-container {
            margin: 0.75rem 0;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.05);
            height: 7px;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
            border-radius: 10px;
            width: 0%;
            transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 0 10px var(--accent-glow);
        }

        /* Class list breakdown */
        .class-breakdown {
            margin-top: 0.75rem;
            max-height: 130px;
            overflow-y: auto;
            padding-right: 5px;
        }

        .class-breakdown::-webkit-scrollbar {
            width: 4px;
        }

        .class-breakdown::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        .breakdown-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            font-size: 0.8rem;
        }

        .breakdown-row:last-child {
            border-bottom: none;
        }

        .breakdown-name {
            color: var(--text-muted);
        }

        .breakdown-pct {
            font-weight: 600;
            color: var(--text-main);
        }

        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Spinner */
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
            display: none;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .btn.loading .spinner {
            display: block;
        }
        .btn.loading span {
            display: none;
        }
    </style>
</head>
<body>
    <div class="glow-bg glow-1"></div>
    <div class="glow-bg glow-2"></div>

    <!-- Menu de Navegação -->
    <nav class="nav-bar">
        <div class="nav-logo">
            <span>YOLO</span> Classifier
        </div>
        <div class="nav-links">
            <a href="/" class="nav-link active" id="navLinkPredict">Testar Classificador</a>
            <a href="/curadoria" class="nav-link" id="navLinkCurate">Painel de Curadoria</a>
        </div>
    </nav>

    <div class="container">
        <div class="header">
            <h1>Classificador de Fertilizantes</h1>
            <p>Inferência YOLO integrada ao pipeline de visão computacional</p>
            <div class="api-badge">
                <div class="badge-dot"></div>
                API Online
            </div>
        </div>

        <!-- Upload Panel -->
        <div class="panel">
            <div class="dropzone-container">
                <!-- Seletor de Modo do Pipeline -->
                <div class="mode-selector-box">
                    <span class="mode-title">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M4 21v-7m0 0V4m0 10h4m8-10v7m0 0v7m0-7h4"/>
                        </svg>
                        Modo do Pipeline
                    </span>
                    <div class="mode-options">
                        <div class="mode-option active" id="optCamera" onclick="selectMode('camera_split')">
                            📷 Câmera Fixa (2 Boxes)
                        </div>
                        <div class="mode-option" id="optCrop" onclick="selectMode('single_crop')">
                            🖼️ Foto / Recorte Único
                        </div>
                    </div>
                </div>

                <div class="dropzone" id="dropzone">
                    <svg viewBox="0 0 24 24">
                        <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                    </svg>
                    <p>Arrastar e soltar arquivo ou <span>procurar imagem</span></p>
                    <p style="font-size: 0.8rem; margin-top: -5px;">Suporta JPG, PNG, WEBP, BMP</p>
                    <input type="file" id="fileInput" accept="image/*">
                </div>

                <div class="preview-box" id="previewBox">
                    <div class="preview-overlay">
                        <span id="fileName">image.jpg</span>
                        <span id="fileSize">0 KB</span>
                    </div>
                </div>

                <button class="btn" id="submitBtn" disabled>
                    <div class="spinner"></div>
                    <span>Executar Pipeline & Classificar</span>
                </button>
            </div>
        </div>

        <!-- Results Panel -->
        <div class="panel">
            <div class="results-container">
                <div class="empty-state" id="emptyState">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                    </svg>
                    <p>Aguardando submissão de imagem para análise</p>
                </div>

                <div id="resultsWrapper" style="display: none;">
                    <div id="cardsContainer" class="results-grid-boxes">
                        <!-- Preenchido dinamicamente para 1 ou 2 boxes -->
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const previewBox = document.getElementById('previewBox');
        const fileNameSpan = document.getElementById('fileName');
        const fileSizeSpan = document.getElementById('fileSize');
        const submitBtn = document.getElementById('submitBtn');
        const emptyState = document.getElementById('emptyState');
        const resultsWrapper = document.getElementById('resultsWrapper');
        const cardsContainer = document.getElementById('cardsContainer');

        let selectedFile = null;
        let selectedMode = 'camera_split';

        function selectMode(mode) {
            selectedMode = mode;
            document.getElementById('optCamera').classList.toggle('active', mode === 'camera_split');
            document.getElementById('optCrop').classList.toggle('active', mode === 'single_crop');
        }

        dropzone.addEventListener('click', function() { fileInput.click(); });

        ['dragenter', 'dragover'].forEach(function(eventName) {
            dropzone.addEventListener(eventName, function(e) {
                e.preventDefault();
                dropzone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(function(eventName) {
            dropzone.addEventListener(eventName, function(e) {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            }, false);
        });

        dropzone.addEventListener('drop', function(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            if (!file.type.match('image.*')) {
                alert('Por favor, selecione apenas arquivos de imagem.');
                return;
            }

            selectedFile = file;
            fileNameSpan.textContent = file.name;
            fileSizeSpan.textContent = (file.size / 1024).toFixed(1) + ' KB';

            const reader = new FileReader();
            reader.onload = function(e) {
                previewBox.style.backgroundImage = `url(${e.target.result})`;
                previewBox.classList.add('active');
            };
            reader.readAsDataURL(file);

            submitBtn.removeAttribute('disabled');
        }

        submitBtn.addEventListener('click', async function() {
            if (!selectedFile) return;

            submitBtn.classList.add('loading');
            submitBtn.setAttribute('disabled', 'true');

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('mode', selectedMode);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Erro na predição');
                }

                const data = await response.json();
                displayResults(data);

            } catch (error) {
                alert('Erro ao submeter imagem: ' + error.message);
            } finally {
                submitBtn.classList.remove('loading');
                submitBtn.removeAttribute('disabled');
            }
        });

        function createBoxCard(title, boxData) {
            const confidencePct = (boxData.confidence * 100).toFixed(1);
            
            let breakdownHtml = '';
            Object.entries(boxData.all_confidences).forEach(function(item) {
                const cls = item[0];
                const conf = item[1];
                const pct = (conf * 100).toFixed(1);
                const isPredicted = cls === boxData.predicted_class;
                const styleName = isPredicted ? 'font-weight: 700; color: var(--accent);' : '';
                const stylePct = isPredicted ? 'color: var(--accent);' : '';

                breakdownHtml += `
                    <div class="breakdown-row">
                        <span class="breakdown-name" style="${styleName}">${cls}</span>
                        <span class="breakdown-pct" style="${stylePct}">${pct}%</span>
                    </div>
                `;
            });

            return `
                <div class="result-card">
                    <div class="result-header">
                        <span class="result-title">${title}</span>
                        <span class="inference-badge">${boxData.inference_time_ms} ms</span>
                    </div>
                    
                    ${boxData.image_base64 ? `<img src="${boxData.image_base64}" class="box-proc-preview" alt="${title} Processado" title="Imagem processada (ROI + CLAHE)">` : ''}

                    <div class="predicted-value">${boxData.predicted_class}</div>
                    
                    <div class="progress-container">
                        <div class="progress-label">
                            <span>Confiança</span>
                            <span>${confidencePct}%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: ${confidencePct}%;"></div>
                        </div>
                    </div>                    
                </div>
            `;
        }

        function displayResults(data) {
            emptyState.style.display = 'none';
            resultsWrapper.style.display = 'block';
            cardsContainer.innerHTML = '';

            if (data.mode === 'camera_split') {
                cardsContainer.style.gridTemplateColumns = '1fr 1fr';
                cardsContainer.innerHTML += createBoxCard('Box Esquerdo', data.boxes.esquerdo);
                cardsContainer.innerHTML += createBoxCard('Box Direito', data.boxes.direito);
            } else {
                cardsContainer.style.gridTemplateColumns = '1fr';
                cardsContainer.innerHTML += createBoxCard('Classificação do Recorte', data);
            }
        }
    </script>
</body>
</html>
"""

CURADORIA_HTML = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Curadoria YOLO Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        {SHARED_CSS}
        
        .curate-grid {
            width: 100%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
            z-index: 10;
        }

        @media (max-width: 768px) {
            .curate-grid {
                grid-template-columns: 1fr;
            }
        }

        .panel-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            color: var(--text-main);
            border-bottom: 1px solid var(--glass-border);
            padding-bottom: 0.75rem;
        }

        .list-panel {
            max-height: 600px;
            overflow-y: auto;
            padding: 1.5rem;
        }

        .list-panel::-webkit-scrollbar {
            width: 4px;
        }
        .list-panel::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }

        .queue-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .queue-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 0.75rem;
            display: flex;
            gap: 0.75rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .queue-item:hover, .queue-item.active {
            border-color: var(--accent);
            background: rgba(6, 182, 212, 0.04);
            transform: translateX(3px);
        }

        .queue-item-thumb {
            width: 50px;
            height: 50px;
            border-radius: 8px;
            background-size: cover;
            background-position: center;
            border: 1px solid var(--glass-border);
            flex-shrink: 0;
        }

        .queue-item-details {
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.15rem;
            overflow: hidden;
            flex-grow: 1;
        }

        .queue-item-name {
            font-size: 0.85rem;
            font-weight: 600;
            white-space: nowrap;
            text-overflow: ellipsis;
            overflow: hidden;
            color: var(--text-main);
        }

        .queue-item-meta {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            font-size: 0.75rem;
        }

        .queue-item-badge {
            background: rgba(79, 70, 229, 0.15);
            border: 1px solid rgba(79, 70, 229, 0.2);
            color: var(--text-main);
            padding: 1px 6px;
            border-radius: 4px;
            font-weight: 600;
        }

        .queue-item-conf {
            color: var(--text-muted);
        }

        .details-panel {
            padding: 2rem;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }

        .curate-img-container {
            width: 100%;
            height: 280px;
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            overflow: hidden;
            margin-bottom: 1.5rem;
            background: #06070a;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.6);
        }

        .curate-img-container img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.5s ease;
        }

        .curate-img-container img:hover {
            transform: scale(1.03);
        }

        .curate-info {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-bottom: 2rem;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 14px;
            padding: 1rem 1.25rem;
        }

        .info-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.4rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.03);
            font-size: 0.9rem;
        }

        .info-row:last-child {
            border-bottom: none;
        }

        .info-label {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        .info-val {
            font-weight: 600;
            color: var(--text-main);
        }

        .predicted-class-badge {
            color: var(--accent);
            font-weight: 700;
        }

        .curate-actions {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .btn-confirm {
            background: linear-gradient(135deg, var(--success) 0%, #065f46 100%);
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
            width: 100%;
        }

        .btn-confirm:hover {
            background: linear-gradient(135deg, #34d399 0%, var(--success) 100%) !important;
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
        }

        .correct-action-group {
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 0.75rem;
        }

        .class-select {
            background: rgba(20, 24, 38, 0.8);
            border: 1px solid var(--glass-border);
            color: var(--text-main);
            border-radius: 12px;
            padding: 0 1rem;
            font-size: 0.95rem;
            outline: none;
            cursor: pointer;
            transition: border-color 0.3s;
        }

        .class-select:focus {
            border-color: var(--primary);
        }

        .class-select option {
            background: #121520;
            color: var(--text-main);
        }

        .btn-correct {
            background: linear-gradient(135deg, var(--accent) 0%, #0891b2 100%);
            box-shadow: 0 4px 15px var(--accent-glow);
        }

        .btn-correct:hover {
            background: linear-gradient(135deg, #22d3ee 0%, var(--accent) 100%) !important;
        }
    </style>
</head>
<body>
    <div class="glow-bg glow-1"></div>
    <div class="glow-bg glow-2"></div>

    <!-- Menu de Navegação -->
    <nav class="nav-bar">
        <div class="nav-logo">
            <span>YOLO</span> Classifier
        </div>
        <div class="nav-links">
            <a href="/" class="nav-link" id="navLinkPredict">Testar Classificador</a>
            <a href="/curadoria" class="nav-link active" id="navLinkCurate">Painel de Curadoria</a>
        </div>
    </nav>

    <div class="curate-grid">
        <!-- Fila de Itens (Esquerda) -->
        <div class="panel list-panel">
            <h2 class="panel-title">Fila Pendente (<span id="queueCount">0</span>)</h2>
            <div class="queue-list" id="queueList">
                <!-- Preenchido dinamicamente por JS -->
            </div>
        </div>

        <!-- Detalhes do Item Selecionado (Direita) -->
        <div class="panel details-panel">
            <div class="empty-state" id="detailsEmpty">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M15 15l-6 6m0 0l-6-6m6 6V9a6 6 0 0112 0v3" />
                </svg>
                <p>Selecione um item da fila de curadoria para iniciar a revisão</p>
            </div>
            
            <div class="details-content" id="detailsContent" style="display: none;">
                <div class="curate-img-container">
                    <img id="curateImg" src="" alt="Imagem para Curadoria">
                </div>
                
                <div class="curate-info">
                    <div class="info-row">
                        <span class="info-label">Nome do Arquivo</span>
                        <span class="info-val" id="infoFilename">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Classe Predita</span>
                        <span class="info-val predicted-class-badge" id="infoPredClass">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Confiança</span>
                        <span class="info-val" id="infoConf">-</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">Hora da Captura</span>
                        <span class="info-val" id="infoDate">-</span>
                    </div>
                </div>
                
                <div class="curate-actions">
                    <button class="btn btn-confirm" id="btnConfirm">
                        <span>Confirmar Predição</span>
                    </button>
                    <div class="correct-action-group">
                        <select class="class-select" id="classSelect">
                            {options_html}
                        </select>
                        <button class="btn btn-correct" id="btnCorrect">
                            <span>Reclassificar</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let queue = [];
        let currentItem = null;

        const queueList = document.getElementById('queueList');
        const queueCount = document.getElementById('queueCount');
        const detailsEmpty = document.getElementById('detailsEmpty');
        const detailsContent = document.getElementById('detailsContent');
        
        const curateImg = document.getElementById('curateImg');
        const infoFilename = document.getElementById('infoFilename');
        const infoPredClass = document.getElementById('infoPredClass');
        const infoConf = document.getElementById('infoConf');
        const infoDate = document.getElementById('infoDate');
        
        const btnConfirm = document.getElementById('btnConfirm');
        const classSelect = document.getElementById('classSelect');
        const btnCorrect = document.getElementById('btnCorrect');

        async function loadQueue() {
            try {
                const response = await fetch('/api/curadoria/pendentes');
                queue = await response.json();
                queueCount.textContent = queue.length;
                
                renderQueue();
            } catch (error) {
                console.error('Erro ao carregar fila:', error);
            }
        }

        function renderQueue() {
            queueList.innerHTML = '';
            
            if (queue.length === 0) {
                queueList.innerHTML = '<div class="empty-state" style="padding: 2rem 0;"><p style="font-size:0.9rem;">Fila vazia!</p></div>';
                detailsEmpty.style.display = 'flex';
                detailsContent.style.display = 'none';
                currentItem = null;
                return;
            }

            queue.forEach(function(item) {
                const el = document.createElement('div');
                el.className = 'queue-item';
                if (currentItem && currentItem.id === item.id) {
                    el.classList.add('active');
                }
                
                const formattedConf = (item.confidence * 100).toFixed(0);
                
                el.innerHTML = `
                    <div class="queue-item-thumb" style="background-image: url(/api/curadoria/imagem/${item.id})"></div>
                    <div class="queue-item-details">
                        <span class="queue-item-name" title="${item.filename}">${item.filename}</span>
                        <div class="queue-item-meta">
                            <span class="queue-item-badge">${item.predicted_class}</span>
                            <span class="queue-item-conf">${formattedConf}% conf.</span>
                        </div>
                    </div>
                `;
                
                el.addEventListener('click', function() { selectItem(item); });
                queueList.appendChild(el);
            });
        }

        function selectItem(item) {
            currentItem = item;
            
            document.querySelectorAll('.queue-item').forEach(function(el) { el.classList.remove('active'); });
            renderQueue(); 
            
            detailsEmpty.style.display = 'none';
            detailsContent.style.display = 'block';

            curateImg.src = `/api/curadoria/imagem/${item.id}`;
            infoFilename.textContent = item.filename;
            infoPredClass.textContent = item.predicted_class;
            infoConf.textContent = (item.confidence * 100).toFixed(1) + '%';
            
            const date = new Date(item.timestamp);
            infoDate.textContent = date.toLocaleString('pt-BR');

            classSelect.value = item.predicted_class;
        }

        btnConfirm.addEventListener('click', function() { submitDecision('confirm', currentItem.predicted_class); });
        btnCorrect.addEventListener('click', function() { submitDecision('correct', classSelect.value); });

        async function submitDecision(action, finalClass) {
            if (!currentItem) return;

            btnConfirm.setAttribute('disabled', 'true');
            btnCorrect.setAttribute('disabled', 'true');

            try {
                const response = await fetch('/api/curadoria/decidir', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        id: currentItem.id,
                        action: action,
                        final_class: finalClass
                    })
                });

                if (!response.ok) {
                    const err = await response.json();
                    throw new Error(err.detail || 'Erro ao processar curadoria');
                }

                await loadQueue();
                
                if (queue.length > 0) {
                    selectItem(queue[0]);
                } else {
                    detailsEmpty.style.display = 'flex';
                    detailsContent.style.display = 'none';
                    currentItem = null;
                }

            } catch (error) {
                alert('Erro na decisão: ' + error.message);
            } finally {
                btnConfirm.removeAttribute('disabled');
                btnCorrect.removeAttribute('disabled');
            }
        }

        loadQueue();
    </script>
</body>
</html>
"""


# --- EXPOSIÇÃO DOS TEMPLATES ---

@app.get("/", response_class=HTMLResponse)
def index():
    """
    Interface Web para testar o classificador.
    """
    return INDEX_HTML.replace("{SHARED_CSS}", SHARED_CSS)


@app.get("/curadoria", response_class=HTMLResponse)
def curadoria_view():
    """
    Interface Web do Painel de Curadoria.
    """
    classes_list = list(model.names.values()) if model else Config.CLASSES
    options_html = ""
    for cls in classes_list:
        options_html += f'<option value="{cls}">{cls}</option>'
        
    return CURADORIA_HTML.replace("{SHARED_CSS}", SHARED_CSS).replace("{options_html}", options_html)
