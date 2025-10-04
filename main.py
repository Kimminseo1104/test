import os
import uuid
import shutil
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx

from PIL import Image, UnidentifiedImageError

import asyncio
import grpc  # AioRpcError 등 사용

# ✅ NestService용 클라이언트 (clova_grpc_client.py는 Nest 버전이어야 함)
from clova_grpc_client import ClovaSpeechClient

# --- 환경 변수 로드 ---
load_dotenv()

app = FastAPI()

# =========================
# 정적 파일 및 루트 라우팅
# =========================
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def read_root():
    """
    루트 접근 시 index.html을 반환.
    파일이 없으면 간단한 안내 HTML을 반환한다.
    """
    index_path = Path("index.html")
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    fallback = """
    <!doctype html>
    <html lang="ko"><head><meta charset="utf-8">
    <title>Server Running</title></head>
    <body style="font-family:sans-serif">
      <h1>FastAPI 서버가 실행 중입니다.</h1>
      <p><code>index.html</code> 파일이 루트에 없어서 기본 페이지를 표시합니다.</p>
      <ul>
        <li><a href="/docs">/docs</a> (Swagger UI)</li>
        <li><a href="/redoc">/redoc</a> (ReDoc)</li>
      </ul>
    </body></html>
    """
    return HTMLResponse(fallback)

@app.get("/favicon.ico")
def favicon():
    """
    favicon 404 소음 방지: 루트에 favicon.ico가 있으면 제공,
    없으면 204 No Content 대체.
    """
    fav = Path("favicon.ico")
    if fav.exists():
        return FileResponse(str(fav))
    return HTMLResponse(status_code=204, content="")

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# =========================
# 업로드 설정
# =========================
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# =========================
# CLOVA Speech API 설정
# =========================
CLOVA_INVOKE_URL = os.getenv("CLOVA_INVOKE_URL")
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY")  # ✅ Nest에서는 이 값만 쓰면 됨
# 아래 두 개는 Nest 경로에서는 사용하지 않음(남겨도 무방)
CLOVA_CLIENT_ID = os.getenv("CLOVA_CLIENT_ID")
CLOVA_CLIENT_SECRET = os.getenv("CLOVA_CLIENT_SECRET")

# =====================================================================================
# 이미지 업로드 API
# =====================================================================================
def _detect_image_format(path: Path) -> str:
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").lower()
            if not fmt:
                raise HTTPException(status_code=400, detail="지원하지 않는 이미지 형식입니다.")
            return "jpg" if fmt == "jpeg" else fmt
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="손상됐거나 이미지가 아닙니다.")

def _save_uploadfile(upload_file: UploadFile, max_bytes: int = MAX_BYTES) -> str:
    if not (upload_file.content_type and upload_file.content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    tmp_name = f"{uuid.uuid4().hex}"
    tmp_path = UPLOAD_DIR / tmp_name

    size = 0
    with tmp_path.open("wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                buffer.close()
                tmp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"파일 용량 제한({MAX_UPLOAD_MB}MB)를 초과했습니다."
                )
            buffer.write(chunk)

    fmt = _detect_image_format(tmp_path)
    final_name = f"{uuid.uuid4().hex}.{fmt}"
    final_path = UPLOAD_DIR / final_name
    shutil.move(str(tmp_path), str(final_path))
    return final_name

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    name = _save_uploadfile(file)
    return JSONResponse({"filename": name, "url": f"/uploads/{name}"})

@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    results = []
    for f in files:
        saved = _save_uploadfile(f)
        results.append({"filename": saved, "url": f"/uploads/{saved}"})
    return JSONResponse({"files": results})

# =====================================================================================
# 저장된 녹음 파일 인식 (CLOVA Speech REST - 필요 시 유지)
# =====================================================================================
class TranscriptionParams(BaseModel):
    """CLOVA Speech API의 params 필드 모델"""
    language: str = Field("ko-KR", description="인식 언어 [ko-KR, en-US, ja, zh-CN, zh-TW]")
    completion: str = Field("async", description="응답 방식 [sync, async]")
    wordAlignment: bool = Field(True, description="단어 정렬 출력 여부")
    fullText: bool = Field(True, description="전체 텍스트 출력 여부")

@app.post("/api/transcribe/upload")
async def transcribe_file_upload(
    media: UploadFile = File(..., description="음성 파일 (MP3, WAV, MP4 등)"),
    language: str = Form("ko-KR", description="인식 언어"),
    completion: str = Form("async", description="응답 방식")
):
    """
    음성 파일을 CLOVA Speech API로 보내 텍스트 변환을 요청 (기본 async).
    성공 시 token 반환.
    """
    if not CLOVA_INVOKE_URL or not CLOVA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CLOVA API 환경 변수가 설정되지 않았습니다.")

    headers = {"X-CLOVASPEECH-API-KEY": CLOVA_SECRET_KEY}

    params_dict = {
        "language": language,
        "completion": completion,
        "wordAlignment": True,
        "fullText": True,
    }
    params_json = json.dumps(params_dict, ensure_ascii=False)

    files = {
        "media": (media.filename, await media.read(), media.content_type),
        "params": (None, params_json, "application/json"),
    }

    clova_url = f"{CLOVA_INVOKE_URL}/recognizer/upload"

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        try:
            resp = await client.post(clova_url, headers=headers, files=files)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"CLOVA API Error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"CLOVA API 요청 실패: {e}")

@app.get("/api/transcribe/status/{token}")
async def transcribe_status(token: str):
    """
    upload API에서 받은 token으로 상태/결과 조회.
    """
    if not CLOVA_INVOKE_URL or not CLOVA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CLOVA API 환경 변수가 설정되지 않았습니다.")

    headers = {"X-CLOVASPEECH-API-KEY": CLOVA_SECRET_KEY}
    clova_url = f"{CLOVA_INVOKE_URL}/recognizer/{token}"

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            resp = await client.get(clova_url, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code,
                                detail=f"CLOVA API Error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"CLOVA API 요청 실패: {e}")

# =====================================================================================
# 실시간 스트리밍 음성 인식 (WebSocket + gRPC - NestService)
# =====================================================================================
@app.websocket("/ws/transcribe/stream")
async def websocket_transcribe_stream(websocket: WebSocket, lang: str = "ko-KR"):
    await websocket.accept()

    grpc_client = None
    audio_queue: asyncio.Queue = asyncio.Queue()
    response_task: asyncio.Task | None = None

    try:
        # ✅ NestService 클라이언트
        if not CLOVA_SECRET_KEY:
            raise RuntimeError("CLOVA_SECRET_KEY 환경변수가 필요합니다.")
        grpc_client = ClovaSpeechClient(secret_key=CLOVA_SECRET_KEY)

        # ✅ 자바 예제와 동일한 CONFIG 구조(camelCase, 중첩)
        config_json = (
            '{'
            f'"transcription":{{"language":"{lang.split("-")[0]}"}}'  # "ko-KR" -> "ko"
            ',"semanticEpd":{"skipEmptyText":false,"useWordEpd":false,"usePeriodEpd":true}'
            '}'
        )

        async def response_handler():
            try:
                async for response in grpc_client.recognize(
                    audio_queue, config_json=config_json, language=lang
                ):
                    contents = getattr(response, "contents", "")
                    if not contents:
                        continue

                    # 서버가 보내는 JSON: {"transcription":{"text":"...","final":true/false}}
                    try:
                        payload = json.loads(contents)
                        if isinstance(payload, dict):
                            tr = payload.get("transcription")
                            if isinstance(tr, dict) and "text" in tr:
                                await websocket.send_json({
                                    "text": tr.get("text", ""),
                                    "is_final": bool(tr.get("final", False)),
                                })
                                continue
                            # 평면 {text,is_final}도 허용
                            if "text" in payload:
                                await websocket.send_json({
                                    "text": payload.get("text", ""),
                                    "is_final": bool(payload.get("is_final", False)),
                                })
                                continue
                        # 알 수 없는 구조면 원문 로그로 보냄
                        await websocket.send_json({"debug": contents})
                    except Exception:
                        await websocket.send_json({"debug": contents})

            except grpc.aio.AioRpcError as e:
                msg = f"gRPC Error: {e.details()} (code: {e.code().name})"
                print(msg)
                try:
                    await websocket.send_json({"error": msg})
                except Exception:
                    pass

        response_task = asyncio.create_task(response_handler())

        # 바이너리 오디오 수신 → 큐로 전달
        while True:
            message = await websocket.receive()
            if "bytes" in message:
                await audio_queue.put(message["bytes"])
            # 텍스트/핑퐁은 무시

    except WebSocketDisconnect:
        await audio_queue.put(None)
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await audio_queue.put(None)
        except Exception:
            pass

        if response_task and not response_task.done():
            response_task.cancel()
            try:
                await response_task
            except asyncio.CancelledError:
                pass

        if grpc_client:
            try:
                await grpc_client.close()
            except Exception:
                pass

        print("Real-time transcription resources cleaned up.")
