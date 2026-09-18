"""JSON-lines worker for the Qwen3-ASR sidecar.

Ported from OpenWhisper's ``services/local_asr/worker.py`` (MIT,
Copyright (c) 2025 Knuckles92), trimmed to the ``qwen_asr`` backend.

Contract (one JSON object per line):
  -> {"id": N, "op": "load", "model_path": "<dir>", "device": "cuda"|"cpu"}
  <- {"id": N, "result": {"device": "cuda"}}
  -> {"id": N, "op": "transcribe", "audio_path": "<raw f32le 16kHz mono file>",
      "language": "auto"|"en"|"ru"|...}
  <- {"id": N, "result": {"text": "...", "segments": [{"text", "start", "end"}]}}
  -> {"id": N, "op": "shutdown"}          (no response; the process exits)
  <- {"id": N|null, "error": "..."}       on any failure
"""

import array
import json
import os
import sys
import traceback


def main():
    # The real OS stdout is duplicated into `protocol` first, then the process
    # stdout fd is redirected to stderr, so library noise (torch/transformers)
    # can never corrupt the JSON channel.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    engine = None
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
        except Exception as error:
            protocol.write(
                json.dumps({"id": None, "error": "Bad request: %s" % error}) + "\n"
            )
            continue

        op = request.get("op")
        try:
            if op == "shutdown":
                return
            if op == "load":
                engine = load(request)
                protocol.write(
                    json.dumps(
                        {
                            "id": request["id"],
                            "result": {"device": request.get("device") or "cpu"},
                        }
                    )
                    + "\n"
                )
            elif op == "transcribe":
                if engine is None:
                    raise RuntimeError("No model loaded")
                result = transcribe(engine, request)
                protocol.write(
                    json.dumps({"id": request["id"], "result": result}, ensure_ascii=True)
                    + "\n"
                )
            else:
                raise RuntimeError("Unknown op: %r" % (op,))
        except Exception:
            traceback.print_exc()
            protocol.write(
                json.dumps(
                    {"id": request.get("id"), "error": format_current_error()},
                    ensure_ascii=True,
                )
                + "\n"
            )


def format_current_error():
    error = sys.exc_info()[1]
    return str(error) if error is not None else "Unknown error"


def load(request):
    import torch
    from qwen_asr import Qwen3ASRModel

    device = request.get("device") or "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable for Qwen. Select CPU or install a compatible NVIDIA driver."
        )
    engine = Qwen3ASRModel.from_pretrained(
        request["model_path"],
        device_map=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        max_inference_batch_size=1,
        max_new_tokens=2048,
    )
    generate = engine.model.generate

    def checked_generate(*args, **kwargs):
        output = generate(*args, **kwargs)
        if (
            output.sequences.shape[1] - kwargs["input_ids"].shape[1]
            >= kwargs["max_new_tokens"]
        ):
            raise RuntimeError("Qwen reached its output limit. Try a shorter audio.")
        return output

    engine.model.generate = checked_generate
    return engine


def transcribe(engine, request):
    import numpy as np

    samples = array.array("f")
    audio_path = request.get("audio_path")
    if audio_path:
        with open(audio_path, "rb") as audio:
            samples.frombytes(audio.read())

    language = request.get("language")
    code = None if language == "auto" else language
    if code in ("en", "en-US"):
        code = "English"
    text = engine.transcribe(
        audio=(np.asarray(samples, dtype=np.float32), 16000), language=code
    )[0].text
    return dict(
        text=text,
        segments=[dict(text=text, start=0.0, end=len(samples) / 16000)] if text else [],
    )


if __name__ == "__main__":
    main()
