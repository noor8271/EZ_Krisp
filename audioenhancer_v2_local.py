import pyaudio
import numpy as np
import torch
from df.enhance import enhance, init_df
import logging
import sys
import os
import argparse
import time
import traceback
import importlib.util

# ---------------------------
# Configure Logging
# ---------------------------
class UTF8StreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__(sys.stdout)
        self.stream = sys.stdout
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
            try:
                sys.stdout.reconfigure(encoding='utf-8')
            except AttributeError:
                self.filter_emojis = True
            else:
                self.filter_emojis = False
        else:
            self.filter_emojis = False

    def emit(self, record):
        msg = self.format(record)
        if self.filter_emojis:
            emoji_map = {
                '📱': '[PHONE]', '🎤': '[MIC]', '🔌': '[CABLE]', '📡': '[ANTENNA]',
                '🔊': '[SPEAKER]', '🎧': '[HEADSET]', '✅': '[CHECK]', '⚠️': '[WARNING]',
                '💡': '[INFO]', '🔍': '[SEARCH]', '🚨': '[ALERT]', '🎛️': '[CONFIG]',
                '📋': '[CLIPBOARD]', '📊': '[STATS]', '🤖': '[AI]', '🎵': '[AUDIO]'
            }
            for emoji, text in emoji_map.items():
                msg = msg.replace(emoji, text)
        self.stream.write(msg + self.terminator)
        self.flush()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        UTF8StreamHandler(),
        logging.FileHandler('audio_enhancer.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


logger.debug(f"Python version: {sys.version}")
logger.debug(f"PyAudio version: {pyaudio.__version__}")
logger.debug(f"NumPy version: {np.__version__}")
logger.debug(f"PyTorch version: {torch.__version__}")
try:
    import df
    logger.debug(f"DeepFilterNet version: {getattr(df, '__version__', 'unknown')}")
except ImportError as e:
    logger.error(f"Failed to import DeepFilterNet: {str(e)}")
    sys.exit(1)

# Check if deepfilternet3 module is available
if importlib.util.find_spec("df.deepfilternet3") is None:
    logger.warning("Module df.deepfilternet3 not found in Python environment. This may cause issues in the executable.")

# ---------------------------
# Command-Line Arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Real-time audio enhancer with AI noise suppression")
parser.add_argument('--bypass-ai', action='store_true', help='Bypass AI processing for debugging')
args = parser.parse_args()
BYPASS_AI_PROCESSING = args.bypass_ai

# ---------------------------
# Smart Channel Conversion Functions
# ---------------------------
def handle_channel_conversion(audio_np, input_channels, output_channels):
    
    try:
        if input_channels == 1 and output_channels == 1:
            return audio_np
        elif input_channels == 1 and output_channels == 2:
            return np.column_stack([audio_np, audio_np])
        elif input_channels == 2 and output_channels == 1:
            stereo_audio = audio_np.reshape(-1, 2)
            return np.mean(stereo_audio, axis=1)
        elif input_channels == 2 and output_channels == 2:
            return audio_np.reshape(-1, 2)
        else:
            logger.warning(f"Unexpected channel config: {input_channels}→{output_channels}")
            return audio_np
    except Exception as e:
        logger.error(f"Channel conversion error: {str(e)}")
        raise

def process_with_ai(audio_np, input_channels, output_channels, model, df_state):
    
    try:
        if input_channels == 1:
            mono_audio = audio_np
        elif input_channels == 2:
            stereo_audio = audio_np.reshape(-1, 2)
            mono_audio = np.mean(stereo_audio, axis=1)
        else:
            mono_audio = audio_np
            logger.warning(f"Unexpected input channels: {input_channels}")

        NOISE_GATE_THRESHOLD = 0.00195  
        input_level = np.abs(mono_audio).mean()
        if input_level < NOISE_GATE_THRESHOLD:
            # uncomment for debugging and check if noise is suppressed during speech(it should not be).
            # logger.debug("Noise gate: Suppressing low-level input")
            return np.zeros_like(mono_audio) if output_channels == 1 else np.zeros((len(mono_audio), 2))

        mono_audio = np.copy(mono_audio)
        audio_tensor = torch.from_numpy(mono_audio).unsqueeze(0)
        with torch.no_grad():
            enhanced_tensor = enhance(model, df_state, audio_tensor, atten_lim_db=40.0)  
        enhanced_mono = enhanced_tensor.squeeze(0).detach().cpu().numpy()

        if output_channels == 1:
            return enhanced_mono
        elif output_channels == 2:
            return np.column_stack([enhanced_mono, enhanced_mono])
        else:
            logger.warning(f"Unexpected output channels: {output_channels}")
            return enhanced_mono
    except Exception as e:
        logger.error(f"AI processing error: {str(e)}")
        raise

# ---------------------------
# Initialize PyAudio
# ---------------------------
try:
    pa = pyaudio.PyAudio()
except Exception as e:
    logger.error(f"Failed to initialize PyAudio: {str(e)}")
    traceback.print_exc(file=open('audio_enhancer.log', 'a'))
    sys.exit(1)

def filter_common_devices(devices, device_type):
    """Filter out duplicate and less useful devices"""
    try:
        filtered = []
        seen_names = set()
        for idx, info in devices:
            name = info.get('name', '').lower()
            if 'microsoft sound mapper' in name or 'primary sound' in name:
                continue
            base_name = name.split('(')[0].strip()
            if base_name in seen_names:
                continue
            seen_names.add(base_name)
            filtered.append((idx, info))
        return filtered
    except Exception as e:
        logger.error(f"Error filtering {device_type} devices: {str(e)}")
        raise

# ---------------------------
# List Audio Devices
# ---------------------------
try:
    logger.info("\n" + "="*60)
    logger.info("📱 MAIN INPUT DEVICES (Microphones):")
    logger.info("="*60)
    all_input_devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get('maxInputChannels') > 0:
            all_input_devices.append((i, info))

    filtered_inputs = filter_common_devices(all_input_devices, "input")
    for i, info in filtered_inputs:
        device_name = info.get('name')
        channels = info.get('maxInputChannels')
        sample_rate = info.get('defaultSampleRate')
        emoji = "🎤" if 'microphone' in device_name.lower() or 'mic' in device_name.lower() else \
                "🎧" if 'headset' in device_name.lower() else \
                "🔌" if 'cable' in device_name.lower() else "📡"
        logger.info(f"{emoji} {i}: {device_name} ({channels} channels, {sample_rate} Hz)")

    logger.info("\n" + "="*60)
    logger.info("🔊 MAIN OUTPUT DEVICES (Speakers/Virtual Cables):")
    logger.info("="*60)
    all_output_devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get('maxOutputChannels') > 0:
            all_output_devices.append((i, info))

    filtered_outputs = filter_common_devices(all_output_devices, "output")
    for i, info in filtered_outputs:
        device_name = info.get('name')
        channels = info.get('maxOutputChannels')
        sample_rate = info.get('defaultSampleRate')
        emoji = "🔊" if 'speaker' in device_name.lower() else \
                "🎧" if 'headphone' in device_name.lower() or 'headset' in device_name.lower() else \
                "🔌" if 'cable' in device_name.lower() else "📡"
        logger.info(f"{emoji} {i}: {device_name} ({channels} channels, {sample_rate} Hz)")

    logger.info(f"\n💡 Showing {len(filtered_inputs)} input and {len(filtered_outputs)} output devices")
    logger.info("🔍 Need to see ALL devices? Modify code to include unfiltered devices.")
except Exception as e:
    logger.error(f"Error listing devices: {str(e)}")
    traceback.print_exc(file=open('audio_enhancer.log', 'a'))
    pa.terminate()
    sys.exit(1)

# ---------------------------
# User Input for Device Selection
# ---------------------------
try:
    if BYPASS_AI_PROCESSING:
        logger.warning("\n🚨 DEBUG MODE: AI PROCESSING IS BYPASSED - Testing raw audio passthrough")
    else:
        logger.info("\n✅ AI PROCESSING ENABLED")

    input_device_index = int(input("\n🎤 Enter INPUT device index: "))
    input_info = pa.get_device_info_by_index(input_device_index)
except (ValueError, IndexError) as e:
    logger.error(f"Invalid input device index: {str(e)}. Falling back to default input device.")
    try:
        input_device_index = pa.get_default_input_device_info()['index']
        input_info = pa.get_device_info_by_index(input_device_index)
        logger.info(f"Using default input device: {input_info['name']}")
    except Exception as e:
        logger.error(f"Failed to select default input device: {str(e)}")
        traceback.print_exc(file=open('audio_enhancer.log', 'a'))
        pa.terminate()
        sys.exit(1)

try:
    output_device_index = int(input("🔊 Enter OUTPUT device index: "))
    output_info = pa.get_device_info_by_index(output_device_index)
except (ValueError, IndexError) as e:
    logger.error(f"Invalid output device index: {str(e)}. Falling back to default output device.")
    try:
        output_device_index = pa.get_default_output_device_info()['index']
        output_info = pa.get_device_info_by_index(output_device_index)
        logger.info(f"Using default output device: {output_info['name']}")
    except Exception as e:
        logger.error(f"Failed to select default output device: {str(e)}")
        traceback.print_exc(file=open('audio_enhancer.log', 'a'))
        pa.terminate()
        sys.exit(1)

try:
    input_sample_rate = int(input_info['defaultSampleRate'])
    output_sample_rate = int(output_info['defaultSampleRate'])
    if input_sample_rate != output_sample_rate:
        logger.warning(f"Sample rate mismatch: Input={input_sample_rate}Hz, Output={output_sample_rate}Hz")
        common_rates = [44100, 48000, 22050, 16000]
        SAMPLE_RATE = None
        for rate in common_rates:
            if rate <= min(input_sample_rate, output_sample_rate):
                SAMPLE_RATE = rate
                break
        if SAMPLE_RATE is None:
            SAMPLE_RATE = min(input_sample_rate, output_sample_rate)
        logger.info(f"Using compromise sample rate: {SAMPLE_RATE}Hz")
    else:
        SAMPLE_RATE = input_sample_rate
        logger.info(f"Sample rates match: {SAMPLE_RATE}Hz")

    latency_target = 0.128
    CHUNK = int(SAMPLE_RATE * latency_target)
    CHUNK = ((CHUNK + 63) // 64) * 64
    actual_latency = CHUNK / SAMPLE_RATE
    logger.info(f"Target latency: {latency_target*1000:.1f}ms")
    logger.info(f"Calculated chunk size: {CHUNK} samples ({actual_latency*1000:.1f}ms actual latency)")

    INPUT_CHANNELS = min(2, input_info['maxInputChannels'])
    OUTPUT_CHANNELS = min(2, output_info['maxOutputChannels'])
    INPUT_CHANNELS = max(1, INPUT_CHANNELS)
    OUTPUT_CHANNELS = max(1, OUTPUT_CHANNELS)
    FORMAT = pyaudio.paFloat32

    if input_info['maxInputChannels'] == 0:
        logger.error("Selected input device has no input channels!")
        pa.terminate()
        sys.exit(1)
    if output_info['maxOutputChannels'] == 0:
        logger.error("Selected output device has no output channels!")
        pa.terminate()
        sys.exit(1)

    logger.info("\n" + "="*60)
    logger.info("🔍 DIAGNOSTIC INFORMATION:")
    logger.info("="*60)
    logger.info(f"Input Device: {input_info['name']}")
    logger.info(f"  Max Input Channels: {input_info['maxInputChannels']}")
    logger.info(f"  Default Sample Rate: {input_info['defaultSampleRate']} Hz")
    logger.info(f"  Default Low Input Latency: {input_info['defaultLowInputLatency']:.3f}s")
    logger.info(f"\nOutput Device: {output_info['name']}")
    logger.info(f"  Max Output Channels: {output_info['maxOutputChannels']}")
    logger.info(f"  Default Sample Rate: {output_info['defaultSampleRate']} Hz")
    logger.info(f"  Default Low Output Latency: {output_info['defaultLowOutputLatency']:.3f}s")
    logger.info(f"\n🎛️ FINAL AUDIO CONFIGURATION:")
    logger.info(f"  Sample Rate: {SAMPLE_RATE} Hz")
    logger.info(f"  Chunk Size: {CHUNK} samples ({actual_latency*1000:.1f}ms latency)")
    logger.info(f"  Input Channels: {INPUT_CHANNELS} (device supports {input_info['maxInputChannels']})")
    logger.info(f"  Output Channels: {OUTPUT_CHANNELS} (device supports {output_info['maxOutputChannels']})")
    logger.info(f"  Audio Format: Float32")

    logger.info(f"\n📋 CHANNEL MAPPING STRATEGY:")
    if INPUT_CHANNELS == 1 and OUTPUT_CHANNELS == 1:
        logger.info("  Strategy: Mono → Mono (direct pass-through)")
    elif INPUT_CHANNELS == 1 and OUTPUT_CHANNELS == 2:
        logger.info("  Strategy: Mono → Stereo (duplicate mono to L+R channels)")
    elif INPUT_CHANNELS == 2 and OUTPUT_CHANNELS == 1:
        logger.info("  Strategy: Stereo → Mono (mix L+R channels)")
    elif INPUT_CHANNELS == 2 and OUTPUT_CHANNELS == 2:
        logger.info("  Strategy: Stereo → Stereo (direct pass-through)")
    if input_info['maxInputChannels'] > 2:
        logger.info(f"  ℹ️ Input device has {input_info['maxInputChannels']} channels but using only {INPUT_CHANNELS}")
    if output_info['maxOutputChannels'] > 2:
        logger.info(f"  ℹ️ Output device has {output_info['maxOutputChannels']} channels but using only {OUTPUT_CHANNELS}")
except Exception as e:
    logger.error(f"Error configuring audio: {str(e)}")
    traceback.print_exc(file=open('audio_enhancer.log', 'a'))
    pa.terminate()
    sys.exit(1)

model, df_state = None, None
if not BYPASS_AI_PROCESSING:
    logger.info("\nLoading DeepFilterNet model...")
    try:
        model_path = os.path.join(os.path.expanduser("~"), ".cache", "DeepFilterNet", "DeepFilterNet", "Cache")
        logger.debug(f"Checking DeepFilterNet model path: {model_path}")
        if os.path.exists(model_path):
            logger.debug(f"Model directory contents: {os.listdir(model_path)}")
        else:
            logger.warning(f"Model directory {model_path} does not exist. May need to download models.")

        model, df_state, _ = init_df()
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("USING GPU POWER!!!!")
        logger.info("Model loaded successfully!")
        logger.debug("Testing enhance function...")
        test_audio = np.random.randn(CHUNK).astype(np.float32)
        test_tensor = torch.from_numpy(test_audio).unsqueeze(0)
        with torch.no_grad():
            enhanced_test = enhance(model, df_state, test_tensor, atten_lim_db=20.0)
        enhanced_np = enhanced_test.squeeze(0).detach().cpu().numpy()
        logger.debug(f"Test successful! Input shape: {test_tensor.shape}, Output shape: {enhanced_test.shape}")
        logger.debug(f"Output numpy shape: {enhanced_np.shape}")
    except Exception as e:
        logger.error(f"Failed to load DeepFilterNet: {str(e)}")
        logger.warning("Falling back to bypass mode due to model loading failure.")
        traceback.print_exc(file=open('audio_enhancer.log', 'a'))
        BYPASS_AI_PROCESSING = True

input_stream, output_stream = None, None
try:
    buffer_multiplier = 4
    input_stream = pa.open(
        format=FORMAT,
        channels=INPUT_CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=input_device_index,
        frames_per_buffer=CHUNK,
        stream_callback=None
    )
    output_stream = pa.open(
        format=FORMAT,
        channels=OUTPUT_CHANNELS,
        rate=SAMPLE_RATE,
        output=True,
        output_device_index=output_device_index,
        frames_per_buffer=CHUNK,
        stream_callback=None
    )
    input_stream.start_stream()
    time.sleep(0.01)
    output_stream.start_stream()
    logger.info(f"Audio streams created with {CHUNK} sample buffers!")
except Exception as e:
    logger.error(f"Error creating audio streams: {str(e)}")
    traceback.print_exc(file=open('audio_enhancer.log', 'a'))
    pa.terminate()
    sys.exit(1)

frame_count = 0
error_count = 0
max_errors = 10
logger.info("🎵 Starting audio processing...")
logger.info("📊 Monitoring for dropouts and buffer issues...")

try:
    time.sleep(0.1)
    while True:
        try:
            data = input_stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.float32)
            expected_input_samples = CHUNK * INPUT_CHANNELS
            if len(audio_np) != expected_input_samples:
                logger.warning(f"Input buffer size mismatch: expected {expected_input_samples}, got {len(audio_np)}")
                if len(audio_np) < expected_input_samples:
                    padding = np.zeros(expected_input_samples - len(audio_np), dtype=np.float32)
                    audio_np = np.concatenate([audio_np, padding])
                else:
                    audio_np = audio_np[:expected_input_samples]

            if frame_count > 10 and np.abs(audio_np).mean() < 1e-6:
                if frame_count % 100 == 0:
                    logger.warning(f"Low audio level detected, possible silence or device issue")

            if BYPASS_AI_PROCESSING:
                enhanced = handle_channel_conversion(audio_np, INPUT_CHANNELS, OUTPUT_CHANNELS)
            else:
                enhanced = process_with_ai(audio_np, INPUT_CHANNELS, OUTPUT_CHANNELS, model, df_state)

            enhanced = np.clip(enhanced.astype(np.float32), -0.9, 0.9)
            if frame_count < 5:
                fade = frame_count / 5.0
                enhanced = enhanced * fade

            expected_output_samples = CHUNK * OUTPUT_CHANNELS
            if enhanced.size != expected_output_samples:
                logger.warning(f"Output size mismatch: expected {expected_output_samples}, got {enhanced.size}")
                if enhanced.size < expected_output_samples:
                    if OUTPUT_CHANNELS == 2:
                        enhanced = enhanced.reshape(-1, OUTPUT_CHANNELS) if enhanced.size % OUTPUT_CHANNELS == 0 else enhanced.flatten()
                        if enhanced.size < expected_output_samples:
                            padding_size = expected_output_samples - enhanced.size
                            padding = np.zeros(padding_size, dtype=np.float32)
                            enhanced = np.concatenate([enhanced.flatten(), padding])
                    else:
                        padding = np.zeros(expected_output_samples - enhanced.size, dtype=np.float32)
                        enhanced = np.concatenate([enhanced, padding])
                else:
                    enhanced = audio_np.flatten()[:expected_output_samples]  
            output_data = np.ascontiguousarray(enhanced).tobytes()
            output_stream.write(output_data, num_frames=CHUNK)
            frame_count += 1

            if frame_count % 200 == 0:
                mode = "BYPASS" if BYPASS_AI_PROCESSING else "AI"
                logger.info(f"[{mode}] Frame {frame_count} | Errors: {error_count}")

        except Exception as inner_e:
            error_count += 1
            logger.error(f"Frame processing error: {str(inner_e)}")
            traceback.print_exc(file=open('audio_enhancer.log', 'a'))
            if error_count > max_errors:
                logger.error("Too many processing errors, stopping...")
                break
            continue

except KeyboardInterrupt:
    logger.info(f"User stopped after {frame_count} frames...")
except Exception as e:
    logger.error(f"Critical error: {str(e)}")
    traceback.print_exc(file=open('audio_enhancer.log', 'a'))
finally:
    try:
        if input_stream and input_stream.is_active():
            input_stream.stop_stream()
            input_stream.close()
        if output_stream and output_stream.is_active():
            output_stream.stop_stream()
            output_stream.close()
        pa.terminate()
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        traceback.print_exc(file=open('audio_enhancer.log', 'a'))

    logger.info(f"\n📊 Session Stats:")
    logger.info(f"  Total frames processed: {frame_count}")
    logger.info(f"  Total errors: {error_count}")
    logger.info(f"  Success rate: {((frame_count-error_count)/max(frame_count,1)*100):.1f}%")
    logger.info("\n💡 Debugging Tips:")
    logger.info("  - Check audio_enhancer.log for detailed error traces.")
    logger.info("  - Try different USB ports or audio devices if you experience dropouts or hum.")
    logger.info("  - Disable system audio enhancements (e.g., Windows 'Loudness Equalization').")
    logger.info("  - Run with --bypass-ai to test without DeepFilterNet.")
    logger.info("  - Verify sample rates in system audio settings (44100 Hz or 48000 Hz recommended).")
    logger.info("  - If DeepFilterNet fails, ensure model files are included in the executable or internet is available.")
    logger.info("  - Adjust NOISE_GATE_THRESHOLD in process_with_ai if hum or dropouts occur.")