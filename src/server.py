"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import asr
import tts

LLM_REPO = "mlx-community/Qwen3.5-4B-MLX-4bit"
SYSTEM_PROMPT = (
    "你是一个友好、健谈的 AI 助手。用户正在通过麦克风和你对话，并通过摄像头展示他们的画面。"
    "如果用户说中文，请用中文回复；如果用户说英文，请用英文回复。"
    "如果用户没有要求的话，请控制输出在几句话以内。不要输出 Emoji。"
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s*")

model = None
processor = None
tts_backend = None
asr_backend = None


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
                    text_parts.append(f"用户说：{transcription}")

                if audio_path and image_path:
                    text_parts.append(
                        "用户同时在展示摄像头画面。如果有相关内容请提及。"
                    )
                elif image_path:
                    text_parts.append("用户正在展示摄像头画面，请描述你看到的。")
                elif not audio_path:
                    text_parts.append(msg.get("text", "你好！"))

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
                    print("Interrupted after LLM, skipping response")
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
    uvicorn.run(app, host="127.0.0.1", port=port)
