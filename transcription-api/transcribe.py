from backends.backend import Transcription
from models import DeviceType
from typing import Optional
import numpy as np
import io
import os

WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "faster-whisper")


def get_backend_class():
    """
    Selects the transcription backend implementation.
    - "faster-whisper" (default): CTranslate2-based, CPU or NVIDIA CUDA only.
    - "openai-whisper": PyTorch-based, works on CPU, NVIDIA CUDA, and AMD/ROCm.
    """
    if WHISPER_BACKEND == "openai-whisper":
        from backends.openaiwhisper import OpenAIWhisperBackend
        return OpenAIWhisperBackend
    from backends.fasterwhisper import FasterWhisperBackend
    return FasterWhisperBackend


def convert_audio(file) -> np.ndarray:
    if WHISPER_BACKEND == "openai-whisper":
        # openai-whisper ships its own ffmpeg-based loader (no faster-whisper dep)
        import whisper
        if isinstance(file, (str, bytes)) or hasattr(file, "read"):
            if hasattr(file, "read"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name
                try:
                    return whisper.load_audio(tmp_path)
                finally:
                    os.remove(tmp_path)
            return whisper.load_audio(file)
    from faster_whisper import decode_audio
    return decode_audio(file, split_stereo=False, sampling_rate=16000)

async def transcribe_from_filename(filename: str,
                                    model_size: int,
                                    language: Optional[str] = None,
                                    device: DeviceType = DeviceType.cpu) -> Transcription:
    
    filepath = os.path.join(os.environ["UPLOAD_DIR"], filename)
    if not os.path.exists(filepath):
        raise RuntimeError(f"file not found in {filepath}")
    audio = convert_audio(filepath)
    return await transcribe_audio(audio, model_size, language, device)

async def transcribe_file(file: io.BytesIO, 
                          model_size: int, 
                          language: Optional[str] = None, 
                          device: DeviceType = DeviceType.cpu) -> Transcription:
    contents = await file.read()  # async read
    if len(contents) < 150 * 1024 * 1024:  # file is smaller than 150MB
            audio = convert_audio(io.BytesIO(contents))
    else:
         # Save the uploaded file temporarily on disk
        with open(file.filename, 'wb') as f:
            f.write(contents)
        # Check if file exists
        if not os.path.exists(file.filename):
            raise RuntimeError(f"file not found in {file.filename}")
        # Corrected to use the function in this file
        audio = convert_audio(file.filename)
        os.remove(file.filename)
    return await transcribe_audio(audio, model_size, language, device)

async def transcribe_audio(audio: np.ndarray, 
                           model_size: int, 
                           language: Optional[str] = None, 
                           device: DeviceType = DeviceType.cpu) -> Transcription:
    
    if language == "auto":
        language = None

    # Load the model
    BackendClass = get_backend_class()
    model = BackendClass(model_size=model_size, device=device)
    model.get_model()
    model.load()
    # Transcribe the file
    return model.transcribe(audio, silent=True, language=language)
