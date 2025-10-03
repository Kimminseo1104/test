# clova_grpc_client.py
import asyncio
from typing import AsyncGenerator, Iterable, Optional, Tuple

import grpc
import clovaspeech_pb2
import clovaspeech_pb2_grpc


class ClovaSpeechClient:
    """
    CLOVA Speech gRPC (v1) 클라이언트
    - 경로 고정: /ncloud.ai.clovaspeech.v1.ClovaSpeechRecognizer/Recognize
    - 첫 메시지: RecognitionConfig (encoding=LINEAR16, 16kHz, language)
    - 이후: audio_content (PCM 16kHz mono S16LE)
    - 인증: x-clovaspeech-api-key (권장) → 실패 시 NCP API GW 키쌍도 시도
    """

    def __init__(
        self,
        client_id: Optional[str],
        client_secret: Optional[str],
        *,
        clova_secret_key: Optional[str] = None,
        host: str = "clovaspeech-gw.ncloud.com",
        port: int = 50051,
    ):
        self.client_id = (client_id or "").strip()
        self.client_secret = (client_secret or "").strip()
        self.clova_secret_key = (clova_secret_key or "").strip()

        target = f"{host}:{port}"
        creds = grpc.ssl_channel_credentials()
        self.channel = grpc.aio.secure_channel(
            target,
            creds,
            options=[
                ("grpc.default_authority", host),
                ("grpc.keepalive_time_ms", 20_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self.stub = clovaspeech_pb2_grpc.ClovaSpeechRecognizerStub(self.channel)

    def _metadata_sets(self) -> Iterable[Tuple[Tuple[str, str], ...]]:
        # 1) 콘솔 Secret Key 1개 (가장 보편적)
        if self.clova_secret_key:
            yield (("x-clovaspeech-api-key", self.clova_secret_key),)

        # 2) NCP API Gateway 키쌍 (환경에 따라 요구될 수 있음)
        if self.client_id and self.client_secret:
            yield (
                ("x-ncp-apigw-api-keyid", self.client_id),
                ("x-ncp-apigw-api-key", self.client_secret),
            )

    async def _request_iter(
        self, audio_q: asyncio.Queue, language: str
    ) -> AsyncGenerator[clovaspeech_pb2.RecognitionRequest, None]:
        # 첫 메시지: config
        cfg = clovaspeech_pb2.RecognitionConfig(
            language=language,
            encoding=clovaspeech_pb2.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            word_alignment=True,
            full_text=True,
        )
        yield clovaspeech_pb2.RecognitionRequest(config=cfg)

        # 이후: audio_content
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                break
            if not chunk:
                continue
            yield clovaspeech_pb2.RecognitionRequest(audio_content=chunk)

    async def recognize(
        self, audio_q: asyncio.Queue, language: str = "ko-KR"
    ) -> AsyncGenerator[clovaspeech_pb2.RecognitionResponse, None]:
        last_error: Optional[Exception] = None

        for md in self._metadata_sets():
            try:
                async for resp in self.stub.Recognize(
                    self._request_iter(audio_q, language),
                    metadata=md,
                    wait_for_ready=True,
                ):
                    yield resp
                return  # 정상 종료
            except grpc.aio.AioRpcError as e:
                last_error = e
                code = e.code().name
                # 인증 문제면 다음 메타데이터 조합으로 시도
                if code in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                    continue
                # 메서드 없음(UNIMPLEMENTED)이나 그 외 오류면 즉시 전달
                raise
            except Exception as e:
                last_error = e
                raise

        if last_error:
            raise last_error
        raise RuntimeError("No valid authentication metadata succeeded.")

    async def close(self):
        await self.channel.close()
