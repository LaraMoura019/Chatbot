from faster_whisper import WhisperModel
import os
from tqdm import tqdm

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
    #model = WhisperModel(modelo_tamanho, device="cpu", compute_type="int8")
    model = WhisperModel(modelo_tamanho, device="cuda", compute_type="float16")

    print("Transcrição iniciada")
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

    # Cria a barra de progresso baseada na duração total do áudio (info.duration)
    with tqdm(total=info.duration, unit="s", desc="A transcrever", bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} seg") as pbar:
        for segmento in segmentos:
            texto_completo += segmento.text + " "
            
            # O tqdm precisa de saber quanto avançou.
            # Nós calculamos a diferença entre onde o segmento acabou e onde a barra está agora.
            pbar.update(segmento.end - pbar.n)
    
    texto_completo = texto_completo.strip()
    
    # Guardar transcrição em ficheiro
    with open(ficheiro_txt, "w", encoding="utf-8") as ficheiro:
        ficheiro.write(texto_completo)

    print("Transcrição concluída")
    
    return texto_completo


# --- Uso ---
#texto = transcricao("./audios/diabetes.mp3", "diabetes.txt", language="en")