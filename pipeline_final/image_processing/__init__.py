"""
Pacote de Processamento de Imagens para Boxes de Fertilizantes.

Módulo de exportação das principais funções, constantes e configurações em português.
"""

from .config import (
    PONTO_DIVISAO_X_PADRAO,
    TAMANHO_IMAGEM_DESTINO,
    POLIGONO_ROI_ESQUERDO_PADRAO,
    POLIGONO_ROI_DIREITO_PADRAO,
    ConfiguracaoTransformacaoBox,
    ConfiguracaoClahe,
    LimiaresQualidade,
    ConfiguracaoPipeline,
    CONFIGURACAO_PADRAO
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

from .quality import (
    calcular_metricas_qualidade
)

from .pipeline import (
    carregar_imagem,
    redimensionar_com_letterbox,
    processar_box,
    processar_imagem_producao,
    salvar_etapas_pipeline
)

__all__ = [
    # Constantes
    "PONTO_DIVISAO_X_PADRAO",
    "TAMANHO_IMAGEM_DESTINO",
    "POLIGONO_ROI_ESQUERDO_PADRAO",
    "POLIGONO_ROI_DIREITO_PADRAO",
    
    # Classes de Configuração
    "ConfiguracaoTransformacaoBox",
    "ConfiguracaoClahe",
    "LimiaresQualidade",
    "ConfiguracaoPipeline",
    "CONFIGURACAO_PADRAO",
    
    # Geometria
    "validar_imagem",
    "dividir_imagem_producao",
    "aplicar_espelhamento",
    "aplicar_rotacao",
    "aplicar_mascara_roi",
    
    # Realce
    "converter_para_escala_cinza",
    "aplicar_clahe",
    
    # Qualidade
    "calcular_metricas_qualidade",
    
    # Pipeline
    "carregar_imagem",
    "redimensionar_com_letterbox",
    "processar_box",
    "processar_imagem_producao",
    "salvar_etapas_pipeline"
]
