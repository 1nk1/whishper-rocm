import os
import math
import uuid
import numpy as np
import whisper  # openai-whisper
import torch
from .backend import Backend, Transcription, Segment


class OpenAIWhisperBackend(Backend):
    """
    Backend based on openai-whisper + PyTorch. Unlike faster-whisper
    (CTranslate2), PyTorch has official ROCm wheels, so this backend is
    what powers GPU acceleration on AMD/ROCm hosts (device="cuda" resolves
    to the HIP/ROCm device under a ROCm-built torch).
    """

    name = "openai-whisper"
    device: str = "cpu"
    model: "whisper.Whisper | None" = None

    def __init__(self, model_size, device: str = "cpu"):
        self.model_size = model_size
        # openai-whisper has no separate .en variants selection quirks;
        # model names match ours 1:1 (tiny, tiny.en, small, ..., large-v3)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.__post_init__()

    def model_path(self) -> str:
        return os.path.join(os.environ["WHISPER_MODELS_DIR"], "openai-whisper")

    def get_model(self) -> None:
        print(f"Downloading model {self.model_size} (openai-whisper backend)...")
        download_root = self.model_path()
        os.makedirs(download_root, exist_ok=True)
        # This both downloads (if missing) and validates the checkpoint.
        whisper.load_model(self.model_size, device="cpu", download_root=download_root)

    def load(self) -> None:
        self.model = whisper.load_model(
            self.model_size, device=self.device, download_root=self.model_path()
        )

    def transcribe(
        self, input: np.ndarray, silent: bool = False, language: str = None
    ) -> Transcription:
        assert self.model is not None
        result = self.model.transcribe(
            input,
            language=language,
            word_timestamps=True,
            fp16=(self.device != "cpu"),
            verbose=False if not silent else None,
        )

        segments: list[Segment] = []
        for seg in result.get("segments", []):
            words = seg.get("words") or []
            segment_extract: Segment = {
                "id": uuid.uuid4().hex,
                "text": seg["text"],
                "start": seg["start"],
                "end": seg["end"],
                "score": round(math.exp(seg.get("avg_logprob", 0.0)), 2),
                "words": [
                    {
                        "start": w["start"],
                        "end": w["end"],
                        "word": w["word"],
                        "score": round(w.get("probability", 0.0), 2),
                    }
                    for w in words
                ],
            }
            segments.append(segment_extract)

        text = " ".join(s["text"] for s in segments)
        text = " ".join(text.strip().split())
        duration = segments[-1]["end"] if segments else 0.0

        transcription: Transcription = {
            "text": text,
            "language": result.get("language", language or ""),
            "duration": duration,
            "segments": segments,
        }
        return transcription
