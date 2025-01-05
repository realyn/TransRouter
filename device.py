import sounddevice as sd


def list_devices():

    devices = sd.query_devices()
    default_input = sd.query_devices(kind='input')
    print(f"\n默认输入设备: {default_input['name']}")
    print(f"支持的采样率: {default_input['default_samplerate']}")

    # 打印可用设备信息，方便调试
    print("\n可用音频设备:")
    print(sd.query_devices())
