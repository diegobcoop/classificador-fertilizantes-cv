"""
Módulo de Configuração para o Pipeline de Processamento de Imagens de Fertilizantes.

Centraliza todas as constantes e parâmetros do pipeline validado:
- Divisão dos boxes (split_x = 1050)
- Transformações geométricas (rotação, deslocamentos, espelhamento, máscara ROI)
- Redimensionamento padronizado em letterbox (640x640)
- Realce e contraste com CLAHE
- Métricas e limiares de qualidade de imagem

Nota para desenvolvedores Pascal/C#:
- As classes abaixo usam `@dataclass`, similares a `records` (Pascal) ou `structs/POCO` (C#).
- Agrupam parâmetros com valores padrão de forma explícita e sem lógica oculta.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# ==============================================================================
# CONSTANTES GLOBAIS DO PIPELINE VALIDADO
# ==============================================================================

# Ponto horizontal de corte da imagem bruta (1920x1080) dividindo box esquerdo e direito
PONTO_DIVISAO_X_PADRAO: int = 1050

# Resolução final de entrada para o modelo YOLO
TAMANHO_IMAGEM_DESTINO: Tuple[int, int] = (640, 640)

# Polígono da ROI do Box Esquerdo (área útil do fertilizante após rotação/alinhamento)
POLIGONO_ROI_ESQUERDO_PADRAO: List[Tuple[int, int]] = [
    (650, 30),
    (976, 22),
    (1042, 684),
    (1038, 968),
    (564, 988),
    (142, 940),
    (135, 758),
    (272, 506),
    (435, 272),
]

# Polígono da ROI do Box Direito (área útil do fertilizante após espelhamento e rotação)
POLIGONO_ROI_DIREITO_PADRAO: List[Tuple[int, int]] = [
    (462, 30),
    (774, 22),
    (852, 746),
    (852, 958),
    (452, 988),
    (34, 920),
    (22, 760),
    (100, 556),
    (256, 302),
]


# ==============================================================================
# CONFIGURAÇÕES ESTRUTURADAS (DATACLASSES)
# ==============================================================================

@dataclass
class ConfiguracaoTransformacaoBox:
    """
    Configurações de transformação geométrica para um box específico (esquerdo ou direito).

    Parâmetros:
    -----------
    espelhar_horizontal : bool
        Se True, espelha horizontalmente o box antes da rotação (cv2.flip com código 1).
    angulo_rotacao : float
        Ângulo em graus para alinhar o box (positivo = anti-horário).
    deslocamento_x : int
        Deslocamento horizontal em pixels aplicado na rotação.
    deslocamento_y : int
        Deslocamento vertical em pixels aplicado na rotação.
    poligono_roi : List[Tuple[int, int]]
        Lista de pontos (x, y) definindo o polígono da área útil do fertilizante.
    valor_preenchimento : int
        Cor para preencher bordas e área externa ao polígono (padrão: 0 = preto).
    """
    espelhar_horizontal: bool = False
    angulo_rotacao: float = 0.0
    deslocamento_x: int = 0
    deslocamento_y: int = 0
    poligono_roi: List[Tuple[int, int]] = field(default_factory=list)
    valor_preenchimento: int = 0


@dataclass
class ConfiguracaoClahe:
    """
    Configurações do filtro CLAHE (Equalização Adaptativa de Histograma).
    """
    limite_contraste: float = 2.0
    tamanho_grade: Tuple[int, int] = (8, 8)


@dataclass
class LimiaresQualidade:
    """
    Limiares estatísticos para diagnóstico de qualidade da imagem.
    """
    brilho_minimo: float = 40.0
    brilho_maximo: float = 220.0
    contraste_minimo: float = 20.0
    nitidez_minima: float = 50.0
    razao_maxima_pixels_escuros: float = 0.40
    razao_maxima_pixels_claros: float = 0.30


@dataclass
class ConfiguracaoPipeline:
    """
    Configuração agregadora de todo o pipeline de processamento de imagens.
    """
    ponto_divisao_x: int = PONTO_DIVISAO_X_PADRAO
    tamanho_destino: Tuple[int, int] = TAMANHO_IMAGEM_DESTINO
    
    # Geometria do Box Esquerdo (validada: sem flip, ângulo 4, offset_x=-10, offset_y=30)
    box_esquerdo: ConfiguracaoTransformacaoBox = field(default_factory=lambda: ConfiguracaoTransformacaoBox(
        espelhar_horizontal=False,
        angulo_rotacao=4.0,
        deslocamento_x=-20,
        deslocamento_y=30,
        poligono_roi=list(POLIGONO_ROI_ESQUERDO_PADRAO),
        valor_preenchimento=0
    ))
    
    # Geometria do Box Direito (validada: com flip horizontal, ângulo 8, offset_x=-15, offset_y=30)
    box_direito: ConfiguracaoTransformacaoBox = field(default_factory=lambda: ConfiguracaoTransformacaoBox(
        espelhar_horizontal=True,
        angulo_rotacao=4.0,
        deslocamento_x=-20,
        deslocamento_y=0,
        poligono_roi=list(POLIGONO_ROI_DIREITO_PADRAO),
        valor_preenchimento=0
    ))
    
    clahe: ConfiguracaoClahe = field(default_factory=ConfiguracaoClahe)
    qualidade: LimiaresQualidade = field(default_factory=LimiaresQualidade)


# Instância padrão pronta para uso imediato
CONFIGURACAO_PADRAO = ConfiguracaoPipeline()
