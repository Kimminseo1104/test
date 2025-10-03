# clova_grpc_client.py
import asyncio
from typing import AsyncGenerator, Iterable, Optional, Tuple

import grpc
import clovaspeech_pb2
import clovaspeech_pb2_grpc

# gRPC 메서드 경로 후보 (UNIMPLEMENTED 나오면 순서대로 시도)
_METHOD_CANDIDATES: Tuple[str, ...] = (
    "/ncloud.ai.clovaspeech.v1.ClovaSpeechRecognizer/Recognize",
    "/ncloud.ai.clovaspeech.external.v1.ClovaSpeechRecognizer/Recognize",
    "/external.v1.ClovaSpeechRecognizer/Recognize",
)

class ClovaSpeechClient:
    """
    CLOVA Speech gRPC 클라이언트
    - 첫 메시지: RecognitionConfig (encoding=LINEAR16, 16kHz 명시)
    - 이후: audio_content (PCM 16kHz mono S16LE)
    - 인증: SecretKey 헤더 또는 NCP API GW 키 쌍을 자동 시도
    - 메서드 경로: v1 / external.v1 등 자동 폴백
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
        # 일부 환경에서 authority/keepalive가 안정성에 도움
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

    def _metadata_sets(self) -> Iterable[Tuple[Tuple[str, str], ...]]:
        # 1) 콘솔 Secret Key (REST에서 쓰던 키) – gRPC에서도 통용되는 경우가 많음
        if self.clova_secret_key:
            yield (("x-clovaspeech-api-key", self.clova_secret_key),)
            yield (("X-CLOVASPEECH-API-KEY", self.clova_secret_key),)  # 대소문자 변형도 시도

        # 2) NCP API Gateway 스타일 (키ID/키)
        if self.client_id and self.client_secret:
            yield (
                ("x-ncp-apigw-api-keyid", self.client_id),
                ("x-ncp-apigw-api-key", self.client_secret),
            )

    async def _request_iter(
        self, audio_q: asyncio.Queue, language: str
    ) -> AsyncGenerator[clovaspeech_pb2.RecognitionRequest, None]:
        # 반드시 첫 요청은 config
        cfg = clovaspeech_pb2.RecognitionConfig(
            language=language,
            # enum 사용 (proto 정의에 맞춤)
            encoding=clovaspeech_pb2.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            word_alignment=True,
            full_text=True,
        )
        yield clovaspeech_pb2.RecognitionRequest(config=cfg)

        # 이후 audio_content 청크
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
            for method in _METHOD_CANDIDATES:
                try:
                    # 기본: 생성된 Stub 경로(v1)로 먼저 시도
                    if method.startswith("/ncloud.ai.clovaspeech.v1"):
                        stub = clovaspeech_pb2_grpc.ClovaSpeechRecognizerStub(self.channel)
                        async for resp in stub.Recognize(
                            self._request_iter(audio_q, language),
                            metadata=md,
                            wait_for_ready=True,
                        ):
                            yield resp
                        return

                    # 패키지가 다른 경우: 경로 직접 호출
                    call = self.channel.stream_stream(
                        method,
                        request_serializer=clovaspeech_pb2.RecognitionRequest.SerializeToString,
                        response_deserializer=clovaspeech_pb2.RecognitionResponse.FromString,
                    )
                    async for resp in call(
                        self._request_iter(audio_q, language),
                        metadata=md,
                        wait_for_ready=True,
                    ):
                        yield resp
                    return

                except grpc.aio.AioRpcError as e:
                    last_error = e
                    code = e.code().name
                    # 메서드 없음 → 다음 경로 후보
                    if code == "UNIMPLEMENTED":
                        continue
                    # 인증 문제 → 다음 메타데이터 후보로
                    if code in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
                        break  # method 루프 탈출 → 다음 md 세트
                    # 그 외(INTERNAL 등) 즉시 전파
                    raise
                except Exception as e:
                    last_error = e
                    raise

        if last_error:
            raise last_error
        raise RuntimeError("No valid metadata/method combination succeeded.")

    async def close(self):
        await self.channel.close()
