import os
import tempfile
import asyncio
import threading
import edge_tts
from faster_whisper import WhisperModel

# ==========================================
# 1. WHISPER (Speech-to-Text / Transcrição)
# ==========================================
_whisper_model = None

def obter_modelo_whisper():
    """Carrega o modelo apenas uma vez e mantém-no na memória"""
    global _whisper_model
    if _whisper_model is None:
        print("A carregar modelo Whisper para a memória da GPU...")
        device = "cuda" 
        compute = "float16"
        _whisper_model = WhisperModel("large-v3", device=device, compute_type=compute)
    return _whisper_model

def transcrever_pergunta(audio_bytes: bytes) -> str:
    """Transcreve o áudio gravado no microfone do browser"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model = obter_modelo_whisper()
        segmentos, _ = model.transcribe(
            tmp_path,
            beam_size=5,
            language="pt",
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        texto = " ".join(s.text for s in segmentos).strip()
    finally:
        os.remove(tmp_path)

    return texto


# ==========================================
# 2. EDGE TTS (Text-to-Speech / Voz da Clara)
# ==========================================
VOZ_PT = "pt-PT-RaquelNeural"

async def _sintetizar_async(texto: str, caminho: str):
    comunicador = edge_tts.Communicate(texto, VOZ_PT)
    await comunicador.save(caminho)

def _correr_em_thread(texto: str, caminho: str):
    """Executa o código assíncrono num loop completamente novo e isolado"""
    asyncio.run(_sintetizar_async(texto, caminho))

def sintetizar_resposta(texto: str) -> bytes:
    """Converte texto em áudio MP3 via Edge TTS usando uma Thread separada"""
    if not texto or not texto.strip():
        return b""

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Iniciamos a geração de voz numa thread separada para não chocar com o Streamlit/uvloop
        thread = threading.Thread(target=_correr_em_thread, args=(texto, tmp_path))
        thread.start()
        thread.join() # Espera que a voz termine de ser gerada
            
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
            
    except Exception as e:
        print(f"Erro na síntese de voz: {e}")
        audio_bytes = b""
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return audio_bytes