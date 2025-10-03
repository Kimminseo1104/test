import os
import uuid
import shutil
import json
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse 
from fastapi.staticfiles import StaticFiles             
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx

from PIL import Image, UnidentifiedImageError

import asyncio
from clova_grpc_client import ClovaSpeechClient

# --- 환경 변수 로드 ---
load_dotenv()

app = FastAPI()

# --- 정적 파일 서비스 설정 (추가) ---
app.mount("/", StaticFiles(directory="."), name="static")

# --- 루트 경로 핸들러 수정 ---
@app.get("/")
def read_root():
    return HTMLResponse(open("index.html","r").read())

# --- 기존 이미지 업로드 설정 ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# --- CLOVA Speech API 설정 ---
CLOVA_INVOKE_URL = os.getenv("CLOVA_INVOKE_URL")
CLOVA_SECRET_KEY = os.getenv("CLOVA_SECRET_KEY")
CLOVA_CLIENT_ID = os.getenv("CLOVA_CLIENT_ID")
CLOVA_CLIENT_SECRET = os.getenv("CLOVA_CLIENT_SECRET")


# =====================================================================================
# 기존 코드: 이미지 업로드 API
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
# 새로운 기능 1: 저장된 녹음 파일 인식 (CLOVA Speech)
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
    음성 파일을 CLOVA Speech API로 보내 텍스트 변환을 요청합니다.
    비동기(async) 요청이 기본이며, 요청 성공 시 작업 상태를 확인할 수 있는 'token'을 반환합니다.
    """
    if not CLOVA_INVOKE_URL or not CLOVA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CLOVA API 환경 변수가 설정되지 않았습니다.")

    # CLOVA API 요청 헤더 및 파라미터 구성
    headers = {
        "X-CLOVASPEECH-API-KEY": CLOVA_SECRET_KEY
    }
    
    # API 가이드에 따라 params는 JSON '문자열'로 전달해야 합니다.
    params_dict = {
        "language": language,
        "completion": completion,
        "wordAlignment": True,
        "fullText": True,
    }
    params_json = json.dumps(params_dict, ensure_ascii=False) # 한글 등을 위해 ensure_ascii=False

    files = {
        "media": (media.filename, await media.read(), media.content_type),
        "params": (None, params_json, "application/json")
    }
    
    clova_url = f"{CLOVA_INVOKE_URL}/recognizer/upload"

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(clova_url, headers=headers, files=files)
            response.raise_for_status() # 2xx 이외의 상태 코드에 대해 예외 발생
            
            # CLOVA API 응답(token 포함)을 클라이언트에 반환
            return response.json()

        except httpx.HTTPStatusError as e:
            # CLOVA API에서 에러 응답이 온 경우
            raise HTTPException(
                status_code=e.response.status_code, 
                detail=f"CLOVA API Error: {e.response.text}"
            )
        except httpx.RequestError as e:
            # 네트워크 오류 등 요청 자체에 문제가 생긴 경우
            raise HTTPException(
                status_code=500,
                detail=f"CLOVA API 요청 실패: {e}"
            )

@app.get("/api/transcribe/status/{token}")
async def transcribe_status(token: str):
    """
    'upload' API에서 받은 token으로 CLOVA Speech의 작업 상태와 결과를 조회합니다.
    """
    if not CLOVA_INVOKE_URL or not CLOVA_SECRET_KEY:
        raise HTTPException(status_code=500, detail="CLOVA API 환경 변수가 설정되지 않았습니다.")

    headers = {
        "X-CLOVASPEECH-API-KEY": CLOVA_SECRET_KEY
    }
    
    clova_url = f"{CLOVA_INVOKE_URL}/recognizer/{token}"
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(clova_url, headers=headers)
            response.raise_for_status()
            
            # CLOVA API 응답(상태, 결과 포함)을 클라이언트에 반환
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code, 
                detail=f"CLOVA API Error: {e.response.text}"
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=500,
                detail=f"CLOVA API 요청 실패: {e}"
            )

# =====================================================================================
# 새로운 기능 2: 실시간 스트리밍 음성 인식 (WebSocket + gRPC)
# =====================================================================================

@app.websocket("/ws/transcribe/stream")
async def websocket_transcribe_stream(websocket: WebSocket, lang: str = "ko-KR"):
    """
    WebSocket을 통해 실시간 음성 데이터를 받아 gRPC로 CLOVA Speech에 전달하고
    인식 결과를 다시 WebSocket으로 클라이언트에게 전송합니다.
    """
    await websocket.accept()
    
    grpc_client = None
    audio_queue = asyncio.Queue()
    
    try:
        # 1. CLOVA gRPC 클라이언트 초기화
        grpc_client = ClovaSpeechClient(CLOVA_CLIENT_ID, CLOVA_CLIENT_SECRET)
        
        # 2. gRPC 서버로부터 결과를 받아 WebSocket으로 전송하는 Task 실행
        async def response_handler():
            try:
                # gRPC recognize 함수는 비동기 제너레이터
                async for response in grpc_client.recognize(audio_queue, language=lang):
                    if response.result and response.result.text:
                        result_text = response.result.text
                        is_final = response.result.final
                        # JSON 형태로 구조화하여 클라이언트에 전송
                        await websocket.send_json({
                            "text": result_text,
                            "is_final": is_final
                        })
            except grpc.aio.AioRpcError as e:
                error_message = f"gRPC Error: {e.details()} (code: {e.code().name})"
                print(error_message)
                await websocket.send_json({"error": error_message})
            except Exception as e:
                print(f"Response handler error: {e}")

        response_task = asyncio.create_task(response_handler())

        # 3. WebSocket으로부터 오디오 데이터를 받아 gRPC 클라이언트로 보내는 루프
        while True:
            audio_data = await websocket.receive_bytes()
            await audio_queue.put(audio_data)

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
        # 스트림 종료를 알리기 위해 큐에 None을 넣음
        await audio_queue.put(None)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # 모든 태스크와 연결을 정리
        if 'response_task' in locals() and not response_task.done():
            response_task.cancel()
        if grpc_client:
            await grpc_client.close()
        print("Real-time transcription resources cleaned up.")
