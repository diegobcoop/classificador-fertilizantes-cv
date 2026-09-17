"""
Módulo de Métricas e Diagnóstico de Qualidade de Imagem.

Calcula métricas estatísticas determinísticas sobre a imagem:
- Brilho médio
- Contraste (Desvio padrão de intensidade)
- Nitidez (Variância do Laplaciano)
- Entropia de Shannon (quantidade de informação de textura)
- Razão de pixels escuros e pixels saturados/claros

Nota para desenvolvedores Pascal/C#:
- Operações matemáticas puras e determinísticas sobre a matriz de pixels.
- Não descarta imagens automaticamente, apenas calcula diagnósticos informativos.
"""

from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from .geometry import validar_imagem
from .config import LimiaresQualidade


def calcular_metricas_qualidade(
    imagem: np.ndarray,
    limiares: Optional[LimiaresQualidade] = None
) -> Dict[str, Any]:
    """
    Calcula métricas quantitativas de qualidade e gera diagnósticos heurísticos.

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem a ser avaliada.
    limiares : LimiaresQualidade, opcional
        Limiares para classificação de alertas. Se None, usa os valores padrão.

    Retorna:
    --------
    Dict[str, Any]
        Dicionário com métricas calculadas ("brilho", "contraste", "nitidez", "entropia", "status", etc.).
    """
    validar_imagem(imagem, "imagem")
    
    if limiares is None:
        limiares = LimiaresQualidade()
    
    # Extrai canal monocromático para estatísticas de luminância
    if imagem.ndim == 3:
        if imagem.shape[2] == 3:
            cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        else:
            cinza = imagem[:, :, 0]
    else:
        cinza = imagem
    
    total_pixels = cinza.size
    if total_pixels == 0:
        raise ValueError("Imagem não contém pixels para cálculo de métricas.")
    
    # 1. Brilho Médio
    brilho = float(np.mean(cinza))
    
    # 2. Contraste (Desvio Padrão)
    contraste = float(np.std(cinza))
    
    # 3. Nitidez (Variância do Laplaciano)
    laplaciano = cv2.Laplacian(cinza, cv2.CV_64F)
    nitidez = float(laplaciano.var())
    
    # 4. Entropia de Shannon (Distribuição de intensidades)
    hist, _ = np.histogram(cinza, bins=256, range=(0, 256))
    dist_prob = hist / total_pixels
    dist_nao_nula = dist_prob[dist_prob > 0]
    entropia = float(-np.sum(dist_nao_nula * np.log2(dist_nao_nula)))
    
    # 5. Proporção de pixels extremos
    pixels_escuros = int(np.sum(cinza < 15))
    pixels_claros = int(np.sum(cinza > 240))
    razao_escuros = float(pixels_escuros / total_pixels)
    razao_claros = float(pixels_claros / total_pixels)
    
    # 6. Alertas Heurísticos
    alertas: List[str] = []
    
    if brilho < limiares.brilho_minimo or razao_escuros > limiares.razao_maxima_pixels_escuros:
        alertas.append("BAIXA_LUMINOSIDADE")
    
    if brilho > limiares.brilho_maximo or razao_claros > limiares.razao_maxima_pixels_claros:
        alertas.append("SUPEREXPOSICAO")
    
    if contraste < limiares.contraste_minimo:
        alertas.append("BAIXO_CONTRASTE")
    
    if nitidez < limiares.nitidez_minima:
        alertas.append("BAIXA_NITIDEZ_OU_OBSTRUCAO")
    
    status = "OK" if len(alertas) == 0 else "|".join(alertas)
    
    return {
        "brilho": round(brilho, 2),
        "contraste": round(contraste, 2),
        "nitidez": round(nitidez, 2),
        "entropia": round(entropia, 2),
        "razao_pixels_escuros": round(razao_escuros, 4),
        "razao_pixels_claros": round(razao_claros, 4),
        "status": status,
        "alertas": alertas
    }
