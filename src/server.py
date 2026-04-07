"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path

import litert_lm
import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import asr
import tts

lang="ee"
if lang=="ee":
    USER_CAMERA_FEEDBACK = "Kasutaja näitab sulle oma kaamerat. Kirjelda, mida sa näed."

    USER_SPOKE_CAMERA = "Kasutaja just rääkis sinuga (heli), näidates samal ajal oma kaamerat (pilti). Vasta lapsele aeglaselt ja selgelt."

    USER_SAID_ = "Kasutaja ütles："

    SYSTEM_PROMPT = (
    "Sa oled sõbralik tehisintellekt, kes räägib 3-aastase lapsega. Ta räägib läbi mikrofoni ja sa näed teda läbi kaamera. Talle meeldivad Põrsas Peppa ja Elsa, seega räägi nendest. Ära kasuta emotikone ega paksu kirja * sümboleid."
    )

    LLM_SKIPPING_RESPONSE = "Katkestatud pärast LLM-i, vastus jäetakse vahele"
else:
    USER_CAMERA_FEEDBACK = "The user is showing you their camera. Describe what you see."

    USER_SPOKE_CAMERA = "The user just spoke to you (audio) while showing their camera (image). Respond slowly and clearly to the kid."

    USER_SAID_ = "User said："

    SYSTEM_PROMPT = (
        "You are a friendly AI talking to a 3 year old. She is  "
        "through a microphone and you see her through the camera. She loves Peppa Pig and Elsa so talk about them. Do not use emoticons or bold * symbols."
    )
    LLM_SKIPPING_RESPONSE = "Interrupted after LLM, skipping response"
#HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
#HF_FILENAME = "gemma-4-E2B-it.litertlm"
HF_REPO = "litert-community/gemma-4-E4B-it-litert-lm"
HF_FILENAME = "gemma-4-E4B-it.litertlm"
LLM_REPO = "mlx-community/Qwen3.5-9B-MLX-4bit"
#LLM_REPO = "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit"
#LLM_REPO = "mlx-community/Qwen3.5-REAP-97B-A10B-4bit"
LLM_REPO = "mlx-community/Qwen3.5-35B-A3B-4bit"


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()
#"You are a friendly, conversational AI assistant. The user is talking to you "
#"You are a friendly pilot flying a plane. The ATC is talking to you "
#"through a microphone and you see the pilot through the camera. Your tail number is November 2 2 foxtrot yankee. Always start with that and skip roger. When atc says negative, they mean you made a mistake, try to correct it"



SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")

model = None
processor = None
tts_backend = None
asr_backend = None

def load_models_gemma():
    global engine, tts_backend
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")

    tts_backend = tts.load()

def load_models():
    global model, processor, tts_backend, asr_backend

    print(f"Loading {LLM_REPO} ...")
    from mlx_vlm import load

    model, processor = load(LLM_REPO)
    print("LLM loaded.")

    tts_backend = tts.load()
    asr_backend = asr.load()


@asynccontextmanager
async def lifespan(app):
    await asyncio.get_event_loop().run_in_executor(None, load_models)
    yield


app = FastAPI(lifespan=lifespan)


def save_temp(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def run_llm(conversation_messages, images=None):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(
        processor,
        model.config,
        conversation_messages,
        num_images=len(images) if images else 0,
        enable_thinking=False,
    )
    result = generate(
        model,
        processor,
        prompt,
        image=images,
        max_tokens=512,
    )
    return result.text


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            audio_path = image_path = None
            interrupted.clear()

            try:
                if msg.get("audio"):
                    audio_path = save_temp(base64.b64decode(msg["audio"]), ".wav")
                if msg.get("image"):
                    image_path = save_temp(base64.b64decode(msg["image"]), ".jpg")

                images = None
                if image_path:
                    images = [image_path]

                text_parts = []
                transcription = None

                if audio_path:
                    print(f"ASR: transcribing {audio_path}...")
                    t_asr = time.time()
                    transcription = asr_backend.transcribe(audio_path)
                    print(f"ASR ({time.time() - t_asr:.2f}s): {transcription!r}")
                    text_parts.append(USER_SAID_ + f"{transcription}")
                #"The user just spoke to you (audio) while showing their camera (image). Respond like you would if you were a real pilot, don't add anything extra."

                if audio_path and image_path:
                    text_parts.append(
                        USER_SPOKE_CAMERA
                    )
                elif image_path:
                    text_parts.append(USER_CAMERA_FEEDBACK)
                elif not audio_path:
                    text_parts.append(msg.get("text", "Hello!"))

                # LLM inference
                user_content = "\n".join(text_parts)
                conversation_history.append({"role": "user", "content": user_content})

                t0 = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: run_llm(conversation_history, images=images)
                )
                llm_time = time.time() - t0

                conversation_history.append({"role": "assistant", "content": response})

                print(f"LLM ({llm_time:.2f}s): {response}")

                if interrupted.is_set():
                    print(LLM_SKIPPING_RESPONSE)
                    continue

                reply = {
                    "type": "text",
                    "text": response,
                    "llm_time": round(llm_time, 2),
                }
                if transcription:
                    reply["transcription"] = transcription
                await ws.send_text(json.dumps(reply))

                if interrupted.is_set():
                    print("Interrupted before TTS, skipping audio")
                    continue

                # Streaming TTS: split into sentences and send chunks progressively
                sentences = split_sentences(response)
                if not sentences:
                    sentences = [response]

                tts_start = time.time()

                # Signal start of audio stream
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_start",
                            "sample_rate": tts_backend.sample_rate,
                            "sentence_count": len(sentences),
                        }
                    )
                )

                for i, sentence in enumerate(sentences):
                    if interrupted.is_set():
                        print(
                            f"Interrupted during TTS (sentence {i + 1}/{len(sentences)})"
                        )
                        break

                    # Generate audio for this sentence
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sentence: tts_backend.generate(s)
                    )

                    if interrupted.is_set():
                        break

                    # Convert to 16-bit PCM and send as base64
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "audio_chunk",
                                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                                "index": i,
                            }
                        )
                    )

                tts_time = time.time() - tts_start
                print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

                if not interrupted.is_set():
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "audio_end",
                                "tts_time": round(tts_time, 2),
                            }
                        )
                    )

            finally:
                for p in [audio_path, image_path]:
                    if p and os.path.exists(p):
                        os.unlink(p)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="localhost", port=port)
