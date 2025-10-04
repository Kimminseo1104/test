# clova_grpc_client.py
import asyncio
from typing import AsyncGenerator, Optional

import grpc

# 반드시 nest.proto로 생성한 파이썬 스텁을 임포트해야 합니다.
#   python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. nest.proto
import nest_pb2
import nest_pb2_grpc


class ClovaSpeechClient:
    """
    NestService gRPC 클라이언트
    - 헤더: authorization: Bearer <secret>
    - 메서드: NestService/recognize (bi-di streaming)
    - 첫 메시지: CONFIG(JSON)
    - 이후: DATA(chunk + extra_contents)
    """

    def __init__(
        self,
        *,
        secret_key: str,
        host: str = "clovaspeech-gw.ncloud.com",
        port: int = 50051,
    ):
        if not secret_key:
            raise ValueError("secret_key가 필요합니다.")
        self.secret_key = secret_key.strip()

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
        self.stub = nest_pb2_grpc.NestServiceStub(self.channel)

        # gRPC(Python)는 메타데이터 키가 소문자여야 함!
        self.metadata = (("authorization", f"Bearer {self.secret_key}"),)

    @staticmethod
    def _lang_to_short(lang: str) -> str:
        """ko-KR -> ko, en-US -> en 등 간단 매핑"""
        if not lang:
            return "ko"
        lang = lang.lower()
        if lang.startswith("ko"):
            return "ko"
        if lang.startswith("en"):
            return "en"
        if lang.startswith("ja"):
            return "ja"
        # 중국어 등은 서버 기대값에 따라 조정
        if lang.startswith("zh-cn"):
            return "zh-CN"
        if lang.startswith("zh-tw"):
            return "zh-TW"
        return lang

    async def _req_iter(
        self, audio_q: asyncio.Queue, config_json: Optional[str], lang: str
    ) -> AsyncGenerator[nest_pb2.NestRequest, None]:
        # 1) CONFIG 먼저(없으면 기본 템플릿 생성)
        if not config_json:
            short = self._lang_to_short(lang)
            config_json = (
                '{'
                f'"transcription":{{"language":"{short}"}},'
                '"semanticEpd":{"skipEmptyText":false,"useWordEpd":false,"usePeriodEpd":true}'
                '}'
            )

        yield nest_pb2.NestRequest(
            type=nest_pb2.CONFIG,
            config=nest_pb2.NestConfig(config=config_json),
        )

        # 2) DATA 스트림
        seq = 0
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                # 종료 신호
                yield nest_pb2.NestRequest(
                    type=nest_pb2.DATA,
                    data=nest_pb2.NestData(
                        chunk=b"",
                        extra_contents=f'{{"seqId": {seq}, "epFlag": true}}',
                    ),
                )
                break

            if not chunk:
                continue

            yield nest_pb2.NestRequest(
                type=nest_pb2.DATA,
                data=nest_pb2.NestData(
                    chunk=chunk,
                    extra_contents=f'{{"seqId": {seq}, "epFlag": false}}',
                ),
            )
            seq += 1

    async def recognize(
        self, audio_q: asyncio.Queue, *, config_json: Optional[str], language: str
    ):
        call = self.stub.recognize(
            self._req_iter(audio_q, config_json, language),
            metadata=self.metadata,
            wait_for_ready=True,
        )
        async for resp in call:
            yield resp

    async def close(self):
        await self.channel.close()
