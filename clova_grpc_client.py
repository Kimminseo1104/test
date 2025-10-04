# clova_grpc_client.py  (NestService용으로 전체 교체)
import asyncio
from typing import AsyncGenerator
import grpc

import nest_pb2
import nest_pb2_grpc


class ClovaSpeechClient:
    """
    NestService 기반 gRPC 클라이언트
    - 엔드포인트: clovaspeech-gw.ncloud.com:50051 (TLS)
    - 인증: metadata {"authorization": "Bearer <SECRET>"}  # 반드시 소문자 key
    - 첫 메시지: RequestType.CONFIG + NestConfig(config=<JSON 문자열>)
    - 이후: RequestType.DATA + NestData(chunk=<bytes>, extra_contents=<JSON 문자열>)
    - 응답: NestResponse.contents (string)
    """

    def __init__(
        self,
        *,
        secret_key: str,
        host: str = "clovaspeech-gw.ncloud.com",
        port: int = 50051,
    ):
        if not secret_key:
            raise ValueError("Secret Key가 비었습니다. CLOVA_SECRET_KEY 환경변수를 확인하세요.")

        target = f"{host}:{port}"
        creds = grpc.ssl_channel_credentials()
        self._channel = grpc.aio.secure_channel(
            target,
            creds,
            options=[
                ("grpc.default_authority", host),
                ("grpc.keepalive_time_ms", 20_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
        )
        self._stub = nest_pb2_grpc.NestServiceStub(self._channel)
        self._metadata = (("authorization", f"Bearer {secret_key}"),)

    async def _req_iter(
        self, audio_q: asyncio.Queue, config_json: str
    ) -> AsyncGenerator[nest_pb2.NestRequest, None]:
        # 1) CONFIG
        yield nest_pb2.NestRequest(
            type=nest_pb2.CONFIG,
            config=nest_pb2.NestConfig(config=config_json),
        )
        # 2) DATA 반복
        seq = 0
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                break
            extra = f'{{"seqId": {seq}, "epFlag": false}}'
            yield nest_pb2.NestRequest(
                type=nest_pb2.DATA,
                data=nest_pb2.NestData(
                    chunk=chunk,
                    extra_contents=extra,
                ),
            )
            seq += 1

    async def recognize(
        self, audio_q: asyncio.Queue, *, config_json: str
    ):
        async for resp in self._stub.recognize(
            self._req_iter(audio_q, config_json),
            metadata=self._metadata,
            wait_for_ready=True,
        ):
            yield resp

    async def close(self):
        await self._channel.close()
