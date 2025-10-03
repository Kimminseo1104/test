import asyncio
import grpc
from typing import AsyncGenerator

# 3단계에서 생성된 파일들을 import
import clovaspeech_pb2
import clovaspeech_pb2_grpc

class ClovaSpeechClient:
    """CLOVA Speech gRPC 클라이언트 로직을 캡슐화한 클래스"""
    def __init__(self, client_id: str, client_secret: str):
        if not client_id or not client_secret:
            raise ValueError("CLOVA API Key가 필요합니다.")
        
        # 실제 gRPC 엔드포인트 주소로 변경해야 합니다.
        self.target_url = "clovaspeech-gw.ncloud.com:50051"
        
        # 인증 메타데이터 생성
        self.metadata = [
            ("x-ncp-apigw-api-keyid", client_id),
            ("x-ncp-apigw-api-key", client_secret)
        ]
        
        # gRPC 채널 생성
        credentials = grpc.ssl_channel_credentials()
        self.channel = grpc.aio.secure_channel(self.target_url, credentials)
        self.stub = clovaspeech_pb2_grpc.ClovaSpeechRecognizerStub(self.channel)

    async def generate_requests(
        self, audio_stream: asyncio.Queue, language: str = "ko-KR"
    ) -> AsyncGenerator[clovaspeech_pb2.RecognitionRequest, None]:
        """오디오 스트림과 설정 정보를 gRPC 요청 스트림으로 변환"""
        
        # 1. 첫 번째 요청: 설정 정보 전송
        config = clovaspeech_pb2.RecognitionConfig(
            language=language,
            encoding=clovaspeech_pb2.RecognitionConfig.AudioEncoding.LINEAR16, # 웹에서 주로 사용하는 PCM 16bit
            sample_rate_hertz=16000, # 샘플링 레이트, 클라이언트와 일치해야 함
            word_alignment=True,
            full_text=True
        )
        yield clovaspeech_pb2.RecognitionRequest(config=config)

        # 2. 두 번째부터: 오디오 데이터 전송
        while True:
            audio_chunk = await audio_stream.get()
            if audio_chunk is None:  # 스트림 종료 신호
                break
            yield clovaspeech_pb2.RecognitionRequest(audio_content=audio_chunk)

    async def recognize(
        self, audio_stream: asyncio.Queue, language: str = "ko-KR"
    ) -> AsyncGenerator[clovaspeech_pb2.RecognitionResponse, None]:
        """gRPC 서버와 양방향 스트리밍 통신을 수행"""
        
        request_generator = self.generate_requests(audio_stream, language)
        
        async for response in self.stub.Recognize(request_generator, metadata=self.metadata):
            yield response
    
    async def close(self):
        """gRPC 채널을 닫습니다."""
        if self.channel:
            await self.channel.close()