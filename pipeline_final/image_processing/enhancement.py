"""
Módulo de Realce e Tratamento Visual (Enhancement).

Contém apenas os filtros validados no pipeline final:
- Conversão para escala de cinza (preservando 3 canais BGR para compatibilidade com redes neurais)
- Realce de contraste adaptativo local (CLAHE)

Nota para desenvolvedores Pascal/C#:
- Todas as funções recebem `np.ndarray` e retornam um NOVO array sem modificar o original.
"""

from typing import Tuple
import cv2
import numpy as np
from .geometry import validar_imagem


def converter_para_escala_cinza(
    imagem: np.ndarray,
    manter_3_canais: bool = True
) -> np.ndarray:
    """
    Converte uma imagem colorida para escala de cinza.

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem BGR ou escala de cinza.
    manter_3_canais : bool
        Se True, replica o canal cinza em 3 canais idênticos (BGR), mantendo shape (H, W, 3).
        Necessário para redes convolucionais e YOLO que esperam 3 canais na entrada.
        Se False, retorna matriz 2D com shape (H, W).

    Retorna:
    --------
    np.ndarray
        Imagem em tons de cinza.
    """
    validar_imagem(imagem, "imagem")
    
    if imagem.ndim == 2:
        cinza = imagem.copy()
    elif imagem.shape[2] == 1:
        cinza = imagem[:, :, 0].copy()
    else:
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    if manter_3_canais:
        return cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    return cinza


def aplicar_clahe(
    imagem: np.ndarray,
    limite_contraste: float = 2.0,
    tamanho_grade: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Aplica equalização de histograma adaptativa limitada por contraste (CLAHE).
    
    - Para imagens em escala de cinza (1 ou 3 canais cinza), equaliza diretamente a intensidade.
    - Para imagens BGR coloridas, converte para LAB e equaliza apenas o canal de luminância (L).

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem BGR ou Grayscale.
    limite_contraste : float
        Limite de contraste local para evitar amplificação de ruído (padrão: 2.0).
    tamanho_grade : Tuple[int, int]
        Tamanho da grade de blocos locais (padrão: 8x8).

    Retorna:
    --------
    np.ndarray
        Imagem com contraste realçado.
    """
    validar_imagem(imagem, "imagem")
    
    clahe_engine = cv2.createCLAHE(clipLimit=limite_contraste, tileGridSize=tamanho_grade)
    
    # 1 canal (matriz 2D ou 3D com 1 canal)
    if imagem.ndim == 2 or (imagem.ndim == 3 and imagem.shape[2] == 1):
        cinza = imagem if imagem.ndim == 2 else imagem[:, :, 0]
        realcada = clahe_engine.apply(cinza)
        if imagem.ndim == 3:
            return realcada[:, :, np.newaxis]
        return realcada
    
    # 3 canais onde R == G == B (escala de cinza de 3 canais)
    if imagem.ndim == 3 and np.array_equal(imagem[:, :, 0], imagem[:, :, 1]):
        cinza_canal = clahe_engine.apply(imagem[:, :, 0])
        return cv2.merge([cinza_canal, cinza_canal, cinza_canal])
    
    # Imagem colorida BGR tradicional: equaliza luminância no espaço LAB
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    enhanced_l = clahe_engine.apply(l_channel)
    merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
