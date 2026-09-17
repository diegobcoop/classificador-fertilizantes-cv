"""
Módulo de Geometria e Transformações Espaciais.

Contém as funções geométricas utilizadas no pipeline validado:
- Validação de formato da imagem
- Divisão da imagem bruta em Box Esquerdo e Box Direito
- Espelhamento horizontal (flip)
- Rotação com translação/deslocamentos (offset)
- Aplicação de máscara poligonal (ROI)

Nota para desenvolvedores Pascal/C#:
- No OpenCV/NumPy, as imagens são matrizes `(altura, largura, canais)`.
- Uma coordenada (x, y) representa:
    - x = posição horizontal (largura, coluna)
    - y = posição vertical (altura, linha)
- O fatiamento (slicing) em matrizes é `imagem[y_min:y_max, x_min:x_max]`.
"""

from typing import List, Tuple
import cv2
import numpy as np


def validar_imagem(imagem: np.ndarray, nome_parametro: str = "imagem") -> None:
    """
    Valida se o array recebido é uma imagem NumPy válida com formato compatível.
    Lança ValueError ou TypeError se for None, vazio ou possuir dimensões incorretas.
    """
    if imagem is None:
        raise ValueError(f"A imagem fornecida em '{nome_parametro}' é None.")
    if not isinstance(imagem, np.ndarray):
        raise TypeError(f"'{nome_parametro}' deve ser um np.ndarray, recebido: {type(imagem).__name__}")
    if imagem.size == 0:
        raise ValueError(f"A imagem fornecida em '{nome_parametro}' está vazia (0 pixels).")
    if imagem.ndim not in (2, 3):
        raise ValueError(
            f"A imagem fornecida em '{nome_parametro}' possui dimensão inválida ({imagem.ndim}). Esperado 2 ou 3."
        )


def dividir_imagem_producao(
    imagem: np.ndarray,
    ponto_divisao_x: int = 1050
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide a imagem bruta da câmera de produção (ex: 1920x1080) em dois boxes: esquerdo e direito.

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem original capturada pela câmera.
    ponto_divisao_x : int
        Coordenada horizontal X onde a imagem será cortada (padrão validado: 1050).

    Retorna:
    --------
    Tuple[np.ndarray, np.ndarray]
        (box_esquerdo, box_direito) como cópias independentes em memória.
    """
    validar_imagem(imagem, "imagem")
    
    altura, largura = imagem.shape[:2]
    
    if not (0 < ponto_divisao_x < largura):
        raise ValueError(
            f"O ponto de divisão ponto_divisao_x={ponto_divisao_x} é inválido para uma imagem de largura {largura}. "
            f"Deve estar entre 1 e {largura - 1}."
        )
    
    box_esquerdo = imagem[:, :ponto_divisao_x].copy()
    box_direito = imagem[:, ponto_divisao_x:].copy()
    
    return box_esquerdo, box_direito


def aplicar_espelhamento(
    imagem: np.ndarray,
    codigo_flip: int = 1
) -> np.ndarray:
    """
    Espelha a imagem.
    
    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem de entrada.
    codigo_flip : int
        1 = espelhamento horizontal (esquerda <-> direita),
        0 = vertical, -1 = ambos. Padrão: 1.
    """
    validar_imagem(imagem, "imagem")
    return cv2.flip(imagem, codigo_flip)


def aplicar_rotacao(
    imagem: np.ndarray,
    angulo: float,
    deslocamento_x: int = 0,
    deslocamento_y: int = 0,
    valor_preenchimento: int = 0,
) -> np.ndarray:
    """
    Gira a imagem ao redor de seu centro geométrico aplicando deslocamento opcional (offset).

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem de entrada.
    angulo : float
        Ângulo em graus. Positivo = anti-horário; negativo = horário.
    deslocamento_x : int
        Deslocamento horizontal (em pixels) aplicado na transformação.
    deslocamento_y : int
        Deslocamento vertical (em pixels) aplicado na transformação.
    valor_preenchimento : int
        Cor/valor dos pixels nas bordas expostas (padrão: 0 = preto).

    Retorna:
    --------
    np.ndarray
        Nova imagem rotacionada e deslocada.
    """
    validar_imagem(imagem, "imagem")
    
    # Se ângulo e deslocamentos forem nulos, retorna cópia sem custo de interpolação
    if abs(angulo) < 1e-4 and deslocamento_x == 0 and deslocamento_y == 0:
        return imagem.copy()
    
    altura, largura = imagem.shape[:2]
    centro_x = largura / 2.0
    centro_y = altura / 2.0
    
    # Matriz 2x3 de rotação afim
    matriz_rotacao = cv2.getRotationMatrix2D(center=(centro_x, centro_y), angle=angulo, scale=1.0)
    matriz_rotacao[0, 2] += deslocamento_x
    matriz_rotacao[1, 2] += deslocamento_y
    
    valor_borda = (
        (valor_preenchimento, valor_preenchimento, valor_preenchimento)
        if imagem.ndim == 3 else valor_preenchimento
    )
    
    rotacionada = cv2.warpAffine(
        imagem,
        matriz_rotacao,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=valor_borda
    )
    
    return rotacionada


def aplicar_mascara_roi(
    imagem: np.ndarray,
    poligono_roi: List[Tuple[int, int]],
    valor_preenchimento: int = 0
) -> np.ndarray:
    """
    Aplica uma máscara poligonal (ROI). Mantém os pixels internos ao polígono
    e preenche toda a área externa com o valor especificado (padrão: 0 = preto).

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem de entrada.
    poligono_roi : List[Tuple[int, int]]
        Lista de pontos (x, y) definindo o polígono da área útil.
        Se a lista contiver menos de 3 pontos, retorna a imagem inalterada.
    valor_preenchimento : int
        Valor/cor de preenchimento do fundo externo (padrão: 0 = preto).

    Retorna:
    --------
    np.ndarray
        Imagem mascarada.
    """
    validar_imagem(imagem, "imagem")
    
    if not poligono_roi or len(poligono_roi) < 3:
        return imagem.copy()
    
    altura, largura = imagem.shape[:2]
    
    # Valida se os pontos do polígono pertencem às dimensões da imagem
    for i, pt in enumerate(poligono_roi):
        if len(pt) != 2:
            raise ValueError(f"Ponto do polígono no índice {i} é inválido: {pt}. Esperado tupla (x, y).")
        x, y = pt
        if x < 0 or y < 0 or x > largura or y > altura:
            raise ValueError(
                f"Ponto do polígono ({x}, {y}) fora dos limites da imagem ({largura}x{altura}) no índice {i}."
            )
    
    # Máscara monocromática binária (0 = fora, 255 = dentro)
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    pts_array = np.array([poligono_roi], dtype=np.int32)
    cv2.fillPoly(mascara, pts_array, 255)
    
    # Imagem de saída com fundo preenchido
    imagem_mascarada = np.full_like(imagem, valor_preenchimento)
    imagem_mascarada[mascara == 255] = imagem[mascara == 255]
    
    return imagem_mascarada
