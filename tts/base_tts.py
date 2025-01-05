from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseTTS(ABC):
    def __init__(self):
        """初始化 TTS 基类"""
        # 采样率和音频格式配置
        self.sample_rate = 16000
        self.bits_per_sample = 16
        self.channels = 1
        
    @abstractmethod
    async def start_session(self):
        """启动 TTS 会话"""
        pass
        
    @abstractmethod
    async def stop_session(self):
        """停止 TTS 会话"""
        pass
        
    @abstractmethod
    async def synthesize_speech(self, text: str, voice_name: Optional[str] = None) -> np.ndarray:
        """
        将文本转换为语音
        
        Args:
            text: 要转换的文本
            voice_name: 语音名称（可选）
            
        Returns:
            numpy.ndarray: PCM音频数据，如果失败返回 None
        """
        pass
        
    @property
    def audio_format(self) -> dict:
        """返回音频格式配置"""
        return {
            'sample_rate': self.sample_rate,
            'bits_per_sample': self.bits_per_sample,
            'channels': self.channels
        }
        
    def set_voice(self, voice_name: str):
        """
        设置语音
        
        Args:
            voice_name: 语音名称
        """
        self.voice_name = voice_name
        
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'stop_session'):
            import asyncio
            asyncio.run(self.stop_session()) 