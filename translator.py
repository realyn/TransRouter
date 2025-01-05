import sounddevice as sd
import numpy as np
import asyncio
from scipy.io import wavfile
import os
from datetime import datetime
from vad_processor import VadProcessor
from gemini_transcriber import GeminiTranscriber
import logging

import device


# 配置日志记录
def setup_logging():
    """设置日志配置"""
    # 创建 logs 目录
    os.makedirs('logs', exist_ok=True)

    # 生成日志文件名（使用时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('logs', f'translator_{timestamp}.log')

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )

    return logging.getLogger('TransRouter')


class AudioTranslator:
    def __init__(self, source_lang="zh-CN"):
        self.logger = setup_logging()

        # 采集音频参数设置
        self.input_sample_rate = 16000  # 修改为16kHz，更适合语音识别
        self.input_channels = 1  # 单声道
        self.input_dtype = np.int16
        self.input_chunk_size = 1600  # 调整块大小为100ms的数据量(16000 * 0.1)
        self.input_buffer = asyncio.Queue()

        # 播放音频参数设置
        self.output_device = "BlackHole 2ch"  # 输出到 Zoom
        self.output_sample_rate = 24000  # 输出到 Zoom
        self.output_channels = 1  # 输出到 Zoom
        self.output_dtype = np.int16  # 输出到 Zoom
        self.output_chunk_size = 2400  # 输出到 Zoom

        # 设备设置
        self.input_device = None  # 使用系统默认麦克风
        self.output_device = None  # "BlackHole 2ch"

        # 初始化 Gemini 转录器
        self.transcriber = GeminiTranscriber()

        # 创建录音和合成音频的保存目录
        self.recordings_dir = "recordings"
        self.synthesis_dir = "synthesis"
        os.makedirs(self.recordings_dir, exist_ok=True)
        os.makedirs(self.synthesis_dir, exist_ok=True)

        # 用于保存音频数据的列表
        self.recording_buffer = []

        # 初始化 VAD 处理器
        # self.vad_processor = VadProcessor(
        #     threshold=0.5,  # VAD 阈值
        #     sampling_rate=self.sample_rate,
        #     min_speech_duration=0.25,  # 最小语音段长度（秒）
        #     silence_duration=0.5  # 静音判断时长（秒）
        # )

        # 创建事件用于控制程序停止
        self.running = True

        # 添加事件循环引用
        self.loop = None

        # 创建播放音频的任务
        self.playback_task = None

    def save_wav(self, audio_data, directory, prefix="", sample_rate=None):
        """保存音频数据为WAV文件"""
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("没有音频数据")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"{prefix}_{timestamp}.wav")

        wavfile.write(filename, sample_rate, audio_data)
        self.logger.info(f"音频已保存: {filename}")
        return filename

    def save_audio(self):
        """保存录音到WAV文件"""
        if not self.recording_buffer:
            self.logger.warning("没有录音数据")
            return

        audio_data = np.concatenate(self.recording_buffer)
        self.save_wav(audio_data, self.recordings_dir,
                      "recording", sample_rate=self.input_sample_rate)
        self.recording_buffer = []

    def audio_record_callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            self.logger.warning(f"状态: {status}")

        if self.loop and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self.input_buffer.put(indata.tobytes()),
                self.loop
            )
        else:
            self.logger.warning("事件循环未运行")

    async def process_audio(self, audio_data):
        """处理音频数据"""
        try:
            speech_segment = audio_data
            if speech_segment is not None:
                await self.transcriber.transcribe_audio(speech_segment)
        except Exception as e:
            self.logger.error(f"处理音频时出错: {e}")

    async def _play_audio(self):
        """播放音频的后台任务"""

        try:
            # 创建异步输出流
            stream = sd.OutputStream(
                samplerate=self.output_sample_rate,
                channels=self.output_channels,
                dtype=self.output_dtype,
                device=self.output_device,
                callback=None,  # 使用异步写入，不需要回调
                finished_callback=None,
                blocksize=self.output_chunk_size
            )

            with stream:
                synthesis_buffer = np.array([], dtype=np.int16)
                while self.running:
                    audio_data = await self.transcriber.audio_in.get()
                    if audio_data is None:
                        self.save_wav(synthesis_buffer, self.synthesis_dir,
                                      "synthesis", sample_rate=self.output_sample_rate)
                        synthesis_buffer = np.array([], dtype=np.int16)
                        continue

                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                    synthesis_buffer = np.append(synthesis_buffer, audio_array)

                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        stream.write,
                        audio_array
                    )
                    # # 等待所有数据写入完成
                    # await asyncio.get_event_loop().run_in_executor(
                    #     None,
                    #     stream.wait
                    # )

            self.logger.info(
                f"播放音频: 采样率={self.output_sample_rate}, 设备={self.output_device}, 数据长度={len(audio_array)}")

        except Exception as e:
            self.logger.error(f"播放音频错误: {e}", exc_info=True)

    async def start_streaming(self):
        """开始音频流处理"""
        try:
            self.loop = asyncio.get_running_loop()
            self.playback_task = asyncio.create_task(self._play_audio())

            stream = sd.InputStream(
                device=self.input_device,
                channels=self.input_channels,
                samplerate=self.input_sample_rate,
                dtype=self.input_dtype,
                blocksize=self.input_chunk_size,
                callback=self.audio_record_callback
            )

            with stream:
                self.logger.info("开始录音...（按 Ctrl+C 停止）")
                while self.running:
                    audio_data = await self.input_buffer.get()
                    await self.process_audio(audio_data)

        except Exception as e:
            self.logger.error(f"发生错误: {e}")
        finally:
            if self.playback_task:
                self.playback_task.cancel()
                try:
                    await self.playback_task
                except asyncio.CancelledError:
                    pass
                self.playback_task = None

            self.loop = None
            self.save_audio()
            if hasattr(self, 'transcriber'):
                await self.transcriber.stop_session()
            if hasattr(self, 'translator'):
                await self.translator.stop_session()

    async def run(self):
        """运行翻译器"""
        try:
            await self.start_streaming()
        except KeyboardInterrupt:
            self.running = False
            self.save_audio()
            self.logger.info("程序已停止")

    async def stop(self):
        """停止翻译器"""
        self.running = False
        if hasattr(self, 'transcriber'):
            await self.transcriber.stop_session()
        if hasattr(self, 'translator'):
            await self.translator.stop_session()


if __name__ == "__main__":
    device.list_devices()
    translator = AudioTranslator()

    async def main():
        try:
            await translator.run()
        except KeyboardInterrupt:
            await translator.stop()
            print("\n程序已停止")

    # 运行主程序
    asyncio.run(main())
