from faster_whisper import WhisperModel
import os

def transcricao(ficheiro_audio, ficheiro_txt, language="en"):
    """
    Transcreve um ficheiro de áudio para texto usando Faster Whisper.
    
    Args:
        ficheiro_audio: caminho para o ficheiro de áudio
        ficheiro_txt: caminho para guardar a transcrição
        language: idioma do áudio ("pt" para português, "en" para inglês)
    
    Returns:
        texto_completo: texto transcrito
    """
    
    # Verificar se o ficheiro existe
    if not os.path.exists(ficheiro_audio):
        raise FileNotFoundError(f"Ficheiro de áudio não encontrado: {ficheiro_audio}")
    
    modelo_tamanho = "large-v3"
    model = WhisperModel(modelo_tamanho, device="cpu", compute_type="int8")
    
    # Fazer a transcrição
    segmentos, info = model.transcribe(
        ficheiro_audio, 
        beam_size=5, # quantos "caminhos" possíveis o modelo explora simultaneamente ao gerar cada palavra
        language=language,     
        condition_on_previous_text=True,  # True melhora coerência em áudios longos (usa o texto já transcrito como contexto para o segmento seguinte)
        vad_filter=True,              # Remove silêncios automaticamente
        vad_parameters=dict(
            min_silence_duration_ms=500
        )
    )
    
    # Construir texto com timestamps 
    texto_completo = ""
    
    for segmento in segmentos:
        texto_completo += segmento.text + " "
    texto_completo = texto_completo.strip()
    
    # Guardar transcrição em ficheiro
    with open(ficheiro_txt, "w", encoding="utf-8") as ficheiro:
        ficheiro.write(texto_completo)
    
    return texto_completo


# --- Uso ---
#texto = transcricao("./audios/diabetes.mp3", "diabetes.txt", language="en")