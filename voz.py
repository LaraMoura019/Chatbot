import os
import tempfile
import asyncio
import edge_tts
import nest_asyncio
from faster_whisper import WhisperModel

# Evita o erro do asyncio no Streamlit
nest_asyncio.apply()

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

def sintetizar_resposta(texto: str) -> bytes:
    """Converte texto em áudio MP3 via Edge TTS"""
    if not texto or not texto.strip():
        return b""

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.run_until_complete(_sintetizar_async(texto, tmp_path))
        else:
            asyncio.run(_sintetizar_async(texto, tmp_path))
            
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
            
    except Exception as e:
        print(f"Erro na síntese de voz (verifique a ligação à internet): {e}")
        audio_bytes = b""
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return audio_bytes