"""
Módulo Principal do Pipeline de Processamento de Imagens.

Implementa a sequência exata e validada de processamento:
1. Divisão da imagem de produção em Box Esquerdo e Direito (split_x = 1050)
2. Para cada box:
   a) Espelhamento horizontal (apenas no Box Direito)
   b) Rotação com offsets específicos (Left: angle=4, x=-10, y=30 / Right: angle=8, x=-15, y=30)
   c) Aplicação de máscara poligonal (ROI)
   d) Redimensionamento para 640x640 com Letterbox (preservando proporção de aspecto)
   e) Conversão para escala de cinza mantendo 3 canais (formato BGR)
   f) Aplicação do filtro CLAHE (limite=2.0, grade=8x8)
3. Retorno dos dois boxes prontos para inferência ou gravação

Nota para desenvolvedores Pascal/C#:
- Fluxo sequencial e determinístico, sem estados globais mutáveis ou efeitos colaterais.
"""

import os
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from .config import (
    ConfiguracaoPipeline,
    ConfiguracaoTransformacaoBox,
    ConfiguracaoClahe,
    CONFIGURACAO_PADRAO,
    TAMANHO_IMAGEM_DESTINO
)
from .geometry import (
    validar_imagem,
    dividir_imagem_producao,
    aplicar_espelhamento,
    aplicar_rotacao,
    aplicar_mascara_roi
)
from .enhancement import (
    converter_para_escala_cinza,
    aplicar_clahe
)


def carregar_imagem(caminho_imagem: str) -> np.ndarray:
    """
    Carrega uma imagem do disco com validação de caminho e integridade.
    Lança FileNotFoundError se o arquivo não existir ou ValueError se não for imagem válida.
    """
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Arquivo de imagem não encontrado: '{caminho_imagem}'")
    
    img = cv2.imread(caminho_imagem)
    if img is None:
        raise ValueError(f"Não foi possível decodificar o arquivo como imagem: '{caminho_imagem}'")
    
    return img


def redimensionar_com_letterbox(
    imagem: np.ndarray,
    tamanho_destino: Tuple[int, int] = TAMANHO_IMAGEM_DESTINO,
    valor_preenchimento: int = 0
) -> np.ndarray:
    """
    Redimensiona a imagem para o tamanho de destino usando letterbox (adicionando barras pretas
    quando necessário) para preservar estritamente a proporção de aspecto (sem distorcer os grãos).

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem de entrada.
    tamanho_destino : Tuple[int, int]
        (largura, altura) de destino. Padrão: (640, 640).
    valor_preenchimento : int
        Valor dos pixels do preenchimento das barras (padrão: 0 = preto).

    Retorna:
    --------
    np.ndarray
        Imagem com formato exato (altura_destino, largura_destino, canais).
    """
    validar_imagem(imagem, "imagem")
    
    largura_alvo, altura_alvo = tamanho_destino
    altura_orig, largura_orig = imagem.shape[:2]

    escala = min(largura_alvo / largura_orig, altura_alvo / altura_orig)
    nova_largura = int(round(largura_orig * escala))
    nova_altura = int(round(altura_orig * escala))

    redimensionada = cv2.resize(
        imagem,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA
    )

    if imagem.ndim == 3:
        saida = np.full(
            (altura_alvo, largura_alvo, imagem.shape[2]),
            valor_preenchimento,
            dtype=imagem.dtype
        )
    else:
        saida = np.full(
            (altura_alvo, largura_alvo),
            valor_preenchimento,
            dtype=imagem.dtype
        )

    deslocamento_x = (largura_alvo - nova_largura) // 2
    deslocamento_y = (altura_alvo - nova_altura) // 2

    saida[
        deslocamento_y:deslocamento_y + nova_altura,
        deslocamento_x:deslocamento_x + nova_largura
    ] = redimensionada

    return saida


def processar_box(
    imagem_box: np.ndarray,
    config_transformacao: ConfiguracaoTransformacaoBox,
    config_clahe: Optional[ConfiguracaoClahe] = None,
    tamanho_destino: Tuple[int, int] = TAMANHO_IMAGEM_DESTINO,
    retornar_etapas: bool = False
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    """
    Executa a cadeia validada de transformações para um box individual.

    Sequência de etapas:
    1. Espelhamento horizontal (se configurado)
    2. Rotação com deslocamento (offset)
    3. Aplicação de máscara poligonal (ROI)
    4. Redimensionamento em Letterbox (640x640)
    5. Conversão para Grayscale (mantendo 3 canais BGR)
    6. Realce de contraste adaptativo CLAHE

    Parâmetros:
    -----------
    imagem_box : np.ndarray
        Imagem recortada do box.
    config_transformacao : ConfiguracaoTransformacaoBox
        Parâmetros geométricos do box (rotação, offsets, polígono ROI, flip).
    config_clahe : ConfiguracaoClahe, opcional
        Parâmetros do CLAHE. Se None, usa padrão (limite=2.0, grade=8x8).
    tamanho_destino : Tuple[int, int]
        Resolução final da imagem (padrão: 640x640).
    retornar_etapas : bool
        Se True, retorna dicionário contendo imagens de cada etapa intermediária.

    Retorna:
    --------
    Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]
        (imagem_processada, etapas_dict_ou_None)
    """
    validar_imagem(imagem_box, "imagem_box")
    
    if config_clahe is None:
        config_clahe = ConfiguracaoClahe()
    
    etapas: Dict[str, np.ndarray] = {}
    atual = imagem_box.copy()
    etapas["00_bruto"] = atual.copy()
    
    # 1. Espelhamento Horizontal (Box Direito)
    if config_transformacao.espelhar_horizontal:
        atual = aplicar_espelhamento(atual, codigo_flip=1)
        etapas["01_espelhado"] = atual.copy()
    
    # 2. Rotação com Offsets
    if (
        abs(config_transformacao.angulo_rotacao) > 1e-4
        or config_transformacao.deslocamento_x != 0
        or config_transformacao.deslocamento_y != 0
    ):
        atual = aplicar_rotacao(
            atual,
            angulo=config_transformacao.angulo_rotacao,
            deslocamento_x=config_transformacao.deslocamento_x,
            deslocamento_y=config_transformacao.deslocamento_y,
            valor_preenchimento=config_transformacao.valor_preenchimento
        )
        etapas["02_rotacionado"] = atual.copy()
    
    # 3. Máscara ROI Poligonal
    if config_transformacao.poligono_roi and len(config_transformacao.poligono_roi) >= 3:
        atual = aplicar_mascara_roi(
            atual,
            poligono_roi=config_transformacao.poligono_roi,
            valor_preenchimento=config_transformacao.valor_preenchimento
        )
        etapas["03_roi"] = atual.copy()
    
    # 4. Redimensionamento Letterbox (640x640)
    atual = redimensionar_com_letterbox(
        atual,
        tamanho_destino=tamanho_destino,
        valor_preenchimento=config_transformacao.valor_preenchimento
    )
    etapas["04_letterbox"] = atual.copy()
    
    # 5. Conversão para Grayscale (3 canais BGR)
    atual = converter_para_escala_cinza(atual, manter_3_canais=True)
    etapas["05_escala_cinza"] = atual.copy()
    
    # 6. CLAHE
    atual = aplicar_clahe(
        atual,
        limite_contraste=config_clahe.limite_contraste,
        tamanho_grade=config_clahe.tamanho_grade
    )
    etapas["06_clahe_final"] = atual.copy()
    
    if retornar_etapas:
        return atual, etapas
    return atual, None


def processar_imagem_producao(
    imagem: np.ndarray,
    config: Optional[ConfiguracaoPipeline] = None,
    retornar_etapas: bool = False
) -> Tuple[Tuple[np.ndarray, np.ndarray], Optional[Dict[str, Dict[str, np.ndarray]]]]:
    """
    Executa o pipeline completo de ponta a ponta em uma imagem bruta da câmera de produção.

    1. Divide a imagem bruta (1920x1080) em box esquerdo e direito (split_x = 1050).
    2. Aplica as transformações completas no Box Esquerdo.
    3. Aplica as transformações completas no Box Direito.

    Parâmetros:
    -----------
    imagem : np.ndarray
        Imagem bruta da câmera fixa.
    config : ConfiguracaoPipeline, opcional
        Configuração completa do pipeline. Se None, usa CONFIGURACAO_PADRAO.
    retornar_etapas : bool
        Se True, retorna dicionários das etapas intermediárias para cada box.

    Retorna:
    --------
    Tuple[Tuple[np.ndarray, np.ndarray], Optional[Dict[str, Dict[str, np.ndarray]]]]
        ((box_esquerdo_processado, box_direito_processado), etapas_ou_None)
    """
    validar_imagem(imagem, "imagem")
    
    if config is None:
        config = CONFIGURACAO_PADRAO
    
    box_esq_bruto, box_dir_bruto = dividir_imagem_producao(
        imagem,
        ponto_divisao_x=config.ponto_divisao_x
    )
    
    box_esq_proc, etapas_esq = processar_box(
        imagem_box=box_esq_bruto,
        config_transformacao=config.box_esquerdo,
        config_clahe=config.clahe,
        tamanho_destino=config.tamanho_destino,
        retornar_etapas=retornar_etapas
    )
    
    box_dir_proc, etapas_dir = processar_box(
        imagem_box=box_dir_bruto,
        config_transformacao=config.box_direito,
        config_clahe=config.clahe,
        tamanho_destino=config.tamanho_destino,
        retornar_etapas=retornar_etapas
    )
    
    etapas_completas = None
    if retornar_etapas:
        etapas_completas = {
            "esquerdo": etapas_esq,
            "direito": etapas_dir
        }
    
    return (box_esq_proc, box_dir_proc), etapas_completas


def salvar_etapas_pipeline(
    etapas: Dict[str, np.ndarray],
    diretorio_saida: str
) -> None:
    """
    Salva em disco as imagens de cada etapa intermediária do pipeline para auditoria.

    Parâmetros:
    -----------
    etapas : Dict[str, np.ndarray]
        Dicionário com o nome da etapa e o array da imagem.
    diretorio_saida : str
        Pasta de destino (criada automaticamente se não existir).
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    
    for nome_etapa, img in etapas.items():
        caminho_arquivo = os.path.join(diretorio_saida, f"{nome_etapa}.jpg")
        cv2.imwrite(caminho_arquivo, img)
