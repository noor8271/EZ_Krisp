
import os
os.environ['DEEPFILTERNET_COMMIT_HASH'] = 'dummy_hash'  


import subprocess
import sys
import pyaudio
import numpy as np
import torch
from df.enhance import enhance, init_df
import logging
import argparse
import time
import traceback
import importlib.util
import threading
import queue
import webview  
import base64
import json

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))  
    return os.path.join(base_path, relative_path)

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


if importlib.util.find_spec("df.deepfilternet3") is None:
    logger.warning("Module df.deepfilternet3 not found. This may cause issues in the executable.")


class MockPopen:
    def __init__(self, *args, **kwargs):
        logger.debug(f"MockPopen called with args: {args}, kwargs: {kwargs}")
        self.stdout = self
        self.args = args
        self.kwargs = kwargs
        self.stderr = self
        self.returncode = 0

    def communicate(self, input=None, timeout=None):
        return (b'dummy_hash', b'')

    def wait(self, timeout=None):
        return 0

    def poll(self):
        return 0

    def read(self, *args, **kwargs):
        return b'dummy_hash'

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

if getattr(sys, 'frozen', False):
    subprocess.Popen = MockPopen
logger.debug(f"Subprocess.Popen mocked: {subprocess.Popen == MockPopen}")
logger.debug(f"DEEPFILTERNET_COMMIT_HASH: {os.environ.get('DEEPFILTERNET_COMMIT_HASH')}")

# ---------------------------
# Command-Line Arguments
# ---------------------------
parser = argparse.ArgumentParser(description="Real-time audio enhancer with AI noise suppression")
parser.add_argument('--bypass-ai', action='store_true', help='Bypass AI processing for debugging')
args = parser.parse_args()
BYPASS_AI_PROCESSING = args.bypass_ai
# ---------------------------
# HTML UI (Embedded)
# ---------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Noise Cancellation Control</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            background: linear-gradient(to bottom right, #1a202c, #2d3748);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: 'Inter', sans-serif;
        }
        .container {
            background: rgba(255, 255, 255, 0.05);
            /* backdrop-filter: blur(10px); Removed to reduce CPU usage */
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            width: 100%;
            max-width: 600px;
            color: white;
        }
        .select {
            background: #2d3748;
            border: 1px solid #4a5568;
            border-radius: 8px;
            padding: 0.5rem;
            width: 100%;
            color: white;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='white'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E");
            background-position: right 0.5rem center;
            background-size: 1.5em;
            background-repeat: no-repeat;
        }
        .select:focus {
            outline: none;
            border-color: #68d391;
            box-shadow: 0 0 0 3px rgba(104, 211, 145, 0.3);
        }
        .toggle {
            position: relative;
            display: inline-block;
            width: 60px;
            height: 34px;
        }
        .toggle input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #4a5568;
            transition: 0.4s;
            border-radius: 34px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 26px;
            width: 26px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: 0.4s;
            border-radius: 50%;
        }
        input:checked + .slider {
            background-color: #68d391;
        }
        input:checked + .slider:before {
            transform: translateX(26px);
        }
        .button {
            background: linear-gradient(to right, #68d391, #48bb78);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            color: white;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(104, 211, 145, 0.4);
        }
        .button:active {
            transform: translateY(0);
        }
        #status {
            transition: color 0.3s;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .pulse {
            animation: pulse 2s infinite; /* Disabled when idle via JavaScript */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-3xl font-bold text-center mb-6">Noise Cancellation Control</h1>
        <div class="space-y-4">
            <div>
                <label class="block text-sm font-medium mb-1">Input Device</label>
                <select id="inputDevice" class="select"></select>
            </div>
            <div>
                <label class="block text-sm font-medium mb-1">Output Device</label>
                <select id="outputDevice" class="select"></select>
            </div>
            <div class="flex items-center space-x-4">
                <label class="text-sm font-medium">Enable Noise Gate</label>
                <label class="toggle">
                    <input type="checkbox" id="noiseGate" checked>
                    <span class="slider"></span>
                </label>
            </div>
            <div class="flex justify-center space-x-4">
                <button class="button" onclick="refreshDevices()">Refresh Devices</button>
                <button class="button" onclick="applyChanges()">Apply Changes</button>
                <button class="button bg-red-600 hover:bg-red-700" onclick="quitApp()">Quit</button>
            </div>
            <div class="text-center">
                <span id="status" class="text-sm">Status: Starting...</span>
            </div>
        </div>
    </div>
    <script>
        let inputDevices = [];
        let outputDevices = [];
        let isActive = false;

        async function init() {
            await refreshDevices();
            updateStatusFromQueue();
        }

        async function refreshDevices() {
            try {
                const devices = await window.pywebview.api.get_devices();
                inputDevices = devices.inputs;
                outputDevices = devices.outputs;
                populateDropdown('inputDevice', inputDevices);
                populateDropdown('outputDevice', outputDevices);
                updateStatusFromQueue();
            } catch (error) {
                updateStatus('Error refreshing devices. Check logs.', true);
            }
        }

        function populateDropdown(id, devices) {
            const select = document.getElementById(id);
            select.innerHTML = '';
            devices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.name;
                option.text = device.name;
                select.appendChild(option);
            });
            if (devices.length > 0) select.value = devices[0].name;
        }

        async function applyChanges() {
            const inputDevice = document.getElementById('inputDevice').value;
            const outputDevice = document.getElementById('outputDevice').value;
            const noiseGate = document.getElementById('noiseGate').checked;
            try {
                await window.pywebview.api.apply_changes(inputDevice, outputDevice, noiseGate);
                updateStatusFromQueue();
                isActive = true;
            } catch (error) {
                updateStatus(`Failed to apply changes: ${error}`, true);
            }
        }

        async function quitApp() {
            await window.pywebview.api.quit_app();
        }

        async function updateStatusFromQueue() {
            try {
                const status = await window.pywebview.api.get_pending_status();
                if (status.message) {
                    updateStatus(status.message, status.isError, status.bypass_ai);
                    isActive = status.message === 'Active';
                }
            } catch (error) {
                updateStatus('Error fetching status. Check logs.', true);
            }
        }

        function updateStatus(message, isError, bypass_ai = false) {
            const statusEl = document.getElementById('status');
            statusEl.textContent = `Status: ${message}${bypass_ai ? ' (Bypass Mode)' : ''}`;
            statusEl.className = isError ? 'text-red-400' : 'text-green-400';
            if (!isError && isActive) {
                statusEl.classList.add('pulse');
            } else {
                statusEl.classList.remove('pulse');
            }
        }

        window.addEventListener('pywebviewready', init);
    </script>
</body>
</html>
"""
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

def process_with_ai(audio_np, input_channels, output_channels, model, df_state, gate_open):
    
    try:
        if input_channels == 1:
            mono_audio = audio_np
        elif input_channels == 2:
            stereo_audio = audio_np.reshape(-1, 2)
            mono_audio = np.mean(stereo_audio, axis=1)
        else:
            mono_audio = audio_np
            logger.warning(f"Unexpected input channels: {input_channels}")
        if gate_open:
            NOISE_GATE_THRESHOLD = 0.00195
            input_level = np.abs(mono_audio).mean()
            if input_level < NOISE_GATE_THRESHOLD:
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
# Device Refresh Function
# ---------------------------
def refresh_devices():
    global pa, stop_event, audio_thread_obj
    try:
        was_running = False
        if audio_thread_obj and audio_thread_obj.is_alive():
            was_running = True
            stop_event.set()
            audio_thread_obj.join(timeout=5.0)
            if audio_thread_obj.is_alive():
                logger.warning("Audio thread did not stop in time during device refresh.")

        try:
            pa.terminate()
        except Exception as e:
            logger.warning(f"Error terminating PyAudio: {str(e)}")
        try:
            pa = pyaudio.PyAudio()
        except Exception as e:
            logger.error(f"Failed to reinitialize PyAudio: {str(e)}")
            state.message_queue.put("Failed to refresh devices: PyAudio error. Check logs.")
            return

        all_input_devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get('maxInputChannels') > 0:
                all_input_devices.append((i, info))

        filtered_inputs = filter_common_devices(all_input_devices, "input")
        state.inputs_list = [(idx, info['name']) for idx, info in filtered_inputs]

        all_output_devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get('maxOutputChannels') > 0:
                all_output_devices.append((i, info))

        filtered_outputs = filter_common_devices(all_output_devices, "output")
        state.outputs_list = [(idx, info['name']) for idx, info in filtered_outputs]

        logger.info(f"Refreshed devices: {len(state.inputs_list)} inputs, {len(state.outputs_list)} outputs")
        state.message_queue.put("Devices refreshed successfully.")

        if was_running and state.input_index is not None and state.output_index is not None:
            try:
                stop_event = threading.Event()
                audio_thread_obj = threading.Thread(target=audio_thread, args=(state, stop_event, model, df_state, pa))
                audio_thread_obj.start()
                logger.info("Restarted audio thread after device refresh.")
                state.message_queue.put("Audio processing resumed after refresh.")
            except Exception as e:
                logger.error(f"Failed to restart audio thread: {str(e)}")
                state.message_queue.put("Failed to resume audio processing. Try applying changes.")
    except Exception as e:
        logger.error(f"Error refreshing devices: {str(e)}")
        state.message_queue.put("Error refreshing devices. Check logs.")

# ---------------------------
# Load DeepFilterNet Model
# ---------------------------
model, df_state = None, None
if not BYPASS_AI_PROCESSING:
    logger.info("\nLoading DeepFilterNet model...")
    try:
        model_path = resource_path('DeepFilterNet3')
        checkpoint_file = os.path.join(model_path, 'checkpoints', 'model_120.ckpt.best')
        config_file = os.path.join(model_path, 'config.ini')
        logger.debug(f"DeepFilterNet model path: {model_path}")
        logger.debug(f"Checkpoint file path: {checkpoint_file} (exists: {os.path.exists(checkpoint_file)})")
        logger.debug(f"Config file path: {config_file} (exists: {os.path.exists(config_file)})")
        logger.debug(f"Temp folder: {os.path.dirname(model_path)}")
        try:
            logger.debug(f"Temp folder contents: {os.listdir(os.path.dirname(model_path))}")
        except Exception as e:
            logger.debug(f"Temp folder contents unavailable: {str(e)}")
        logger.debug(f"Model directory: {model_path} (exists: {os.path.exists(model_path)})")
        try:
            logger.debug(f"Model directory contents: {os.listdir(model_path)}")
        except Exception as e:
            logger.debug(f"Model directory contents unavailable: {str(e)}")
        try:
            checkpoints_dir = os.path.join(model_path, 'checkpoints')
            logger.debug(f"Checkpoints directory: {checkpoints_dir} (exists: {os.path.exists(checkpoints_dir)})")
            logger.debug(f"Checkpoints directory contents: {os.listdir(checkpoints_dir) if os.path.exists(checkpoints_dir) else 'N/A'}")
        except Exception as e:
            logger.debug(f"Checkpoints directory contents unavailable: {str(e)}")

        if not os.path.exists(checkpoint_file):
            logger.error(f"Checkpoint file not found: {checkpoint_file}")
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint_file}")
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            raise FileNotFoundError(f"Config missing: {config_file}")

        logger.debug("Attempting to initialize DeepFilterNet model...")
        model, df_state, _ = init_df(model_base_dir=model_path)
        logger.debug("DeepFilterNet model initialized successfully")
        
        if torch.cuda.is_available():
            logger.debug("CUDA is available, moving model to GPU")
            model = model.cuda()
            logger.info("USING GPU POWER!!!!")
        logger.info("Model loaded successfully!")
        logger.debug("Testing enhance function...")
        test_audio = np.random.randn(1024).astype(np.float32)
        test_tensor = torch.from_numpy(test_audio).unsqueeze(0)
        with torch.no_grad():
            enhanced_test = enhance(model, df_state, test_tensor, atten_lim_db=40.0)
        enhanced_np = enhanced_test.squeeze(0).detach().cpu().numpy()
        logger.debug(f"Test successful! Input shape: {test_tensor.shape}, Output shape: {enhanced_test.shape}")
        logger.debug(f"Output numpy shape: {enhanced_np.shape}")
    except Exception as e:
        logger.error(f"Failed to load DeepFilterNet: {str(e)}")
        logger.error("DeepFilterNet model loading failed. Check paths and dependencies in the EXE environment.")
        traceback.print_exc(file=open('audio_enhancer.log', 'a'))
        if getattr(sys, 'frozen', False):
            logger.error("Cannot proceed in EXE mode without DeepFilterNet. Exiting.")
            sys.exit(1)
        else:
            logger.warning("Falling back to bypass mode in development environment.")
            BYPASS_AI_PROCESSING = True
# ---------------------------
# Shared Audio State Class
# ---------------------------
class AudioState:
    def __init__(self):
        self.lock = threading.Lock()
        self.message_queue = queue.Queue()
        self.input_index = None
        self.output_index = None
        self.gate_open = True
        self.status = "Starting..."
        self.inputs_list = []
        self.outputs_list = []
        self.bypass_ai = BYPASS_AI_PROCESSING
        self.sample_rate = None
        self.chunk = None
        self.input_channels = None
        self.output_channels = None
        self.format = pyaudio.paFloat32
        self.frame_count = 0
        self.error_count = 0

# Global state
state = AudioState()
audio_thread_obj = None
stop_event = None

# ---------------------------
# Configure Audio Params
# ---------------------------
def configure_audio_params(input_index, output_index):
    try:
        input_info = pa.get_device_info_by_index(input_index)
        output_info = pa.get_device_info_by_index(output_index)

        input_sample_rate = int(input_info['defaultSampleRate'])
        output_sample_rate = int(output_info['defaultSampleRate'])
        common_rates = [44100, 48000, 22050, 16000]
        sample_rate = None
        for rate in common_rates:
            if rate <= min(input_sample_rate, output_sample_rate):
                sample_rate = rate
                break
        if sample_rate is None:
            sample_rate = min(input_sample_rate, output_sample_rate)
        if input_sample_rate != output_sample_rate:
            logger.warning(f"Sample rate mismatch: Input={input_sample_rate}Hz, Output={output_sample_rate}Hz")
            logger.info(f"Using compromise sample rate: {sample_rate}Hz")
        else:
            logger.info(f"Sample rates match: {sample_rate}Hz")

        latency_target = 0.128
        chunk = int(sample_rate * latency_target)
        chunk = ((chunk + 63) // 64) * 64
        actual_latency = chunk / sample_rate
        logger.info(f"Target latency: {latency_target*1000:.1f}ms")
        logger.info(f"Calculated chunk size: {chunk} samples ({actual_latency*1000:.1f}ms latency)")
        input_channels = min(2, input_info['maxInputChannels'])
        output_channels = min(2, output_info['maxOutputChannels'])
        input_channels = max(1, input_channels)
        output_channels = max(1, output_channels)

        if input_info['maxInputChannels'] == 0:
            raise ValueError("Selected input device has no input channels")
        if output_info['maxOutputChannels'] == 0:
            raise ValueError("Selected output device has no output channels")

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
        logger.info(f"  Default Low Output Latency: {output_info.get('defaultLowOutputLatency', 0):.3f}s")
        logger.info(f"\n🎛️ FINAL AUDIO CONFIGURATION:")
        logger.info(f"  Sample Rate: {sample_rate} Hz")
        logger.info(f"  Chunk Size: {chunk} samples ({actual_latency*1000:.1f}ms latency)")
        logger.info(f"  Input Channels: {input_channels} (device supports {input_info['maxInputChannels']})")
        logger.info(f"  Output Channels: {output_channels} (device supports {output_info['maxOutputChannels']})")
        logger.info(f"  Audio Format: Float32")

        logger.info(f"\n📋 CHANNEL MAPPING STRATEGY:")
        if input_channels == 1 and output_channels == 1:
            logger.info("  Strategy: Mono → Mono (direct pass-through)")
        elif input_channels == 1 and output_channels == 2:
            logger.info("  Strategy: Mono → Stereo (duplicate mono to L+R channels)")
        elif input_channels == 2 and output_channels == 1:
            logger.info("  Strategy: Stereo → Mono (mix L+R channels)")
        elif input_channels == 2 and output_channels == 2:
            logger.info("  Strategy: Stereo → Stereo (direct pass-through)")
        if input_info['maxInputChannels'] > 2:
            logger.info(f"  ℹ️ Input device has {input_info['maxInputChannels']} channels but using only {input_channels}")
        if output_info['maxOutputChannels'] > 2:
            logger.info(f"  ℹ️ Input device has {output_info['maxOutputChannels']} channels but using only {output_channels}")

        return sample_rate, chunk, input_channels, output_channels
    except Exception as e:
        logger.error(f"Error configuring audio params: {str(e)}")
        raise

# ---------------------------
# Audio Processing Thread
# ---------------------------
def audio_thread(state, stop_event, model, df_state, pa):
    with state.lock:
        input_index = state.input_index
        output_index = state.output_index
        bypass_ai = state.bypass_ai
        sample_rate = state.sample_rate
        chunk = state.chunk
        input_channels = state.input_channels
        output_channels = state.output_channels
        audio_format = state.format
        state.status = "Active"
        state.frame_count = 0
        state.error_count = 0

    input_stream = None
    output_stream = None
    try:
        input_stream = pa.open(
            format=audio_format,
            channels=input_channels,
            rate=sample_rate,
            input=True,
            input_device_index=input_index,
            frames_per_buffer=chunk,
            stream_callback=None
        )
        output_stream = pa.open(
            format=audio_format,
            channels=output_channels,
            rate=sample_rate,
            output=True,
            output_device_index=output_index,
            frames_per_buffer=chunk,
            stream_callback=None
        )
        input_stream.start_stream()
        time.sleep(0.01)
        output_stream.start_stream()
        logger.info(f"Audio streams started with chunk {chunk}")
        state.message_queue.put({'message': 'Active', 'isError': False, 'bypass_ai': bypass_ai})

        max_errors = 10
        while not stop_event.is_set():

            try:
                data = input_stream.read(chunk, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.float32)
                expected_input_samples = chunk * input_channels
                if len(audio_np) != expected_input_samples:
                    logger.warning(f"Input buffer size mismatch: expected {expected_input_samples}, got {len(audio_np)}")
                    if len(audio_np) < expected_input_samples:
                        padding = np.zeros(expected_input_samples - len(audio_np), dtype=np.float32)
                        audio_np = np.concatenate([audio_np, padding])
                    else:
                        audio_np = audio_np[:expected_input_samples]

                if state.frame_count > 10 and np.abs(audio_np).mean() < 1e-6:
                    if state.frame_count % 100 == 0:
                        logger.warning(f"Low audio level detected, possible silence or device issue")
                    time.sleep(0.001)

                with state.lock:
                    gate_open = state.gate_open

                if bypass_ai:
                    enhanced = handle_channel_conversion(audio_np, input_channels, output_channels)
                else:
                    enhanced = process_with_ai(audio_np, input_channels, output_channels, model, df_state, gate_open)

                enhanced = np.clip(enhanced.astype(np.float32), -0.9, 0.9)
                if state.frame_count < 5:
                    fade = state.frame_count / 5.0
                    enhanced = enhanced * fade

                expected_output_samples = chunk * output_channels
                if enhanced.size != expected_output_samples:
                    logger.warning(f"Output size mismatch: expected {expected_output_samples}, got {enhanced.size}")
                    if enhanced.size < expected_output_samples:
                        if output_channels == 2:
                            enhanced = enhanced.reshape(-1, output_channels) if enhanced.size % output_channels == 0 else enhanced.flatten()
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
                output_stream.write(output_data, num_frames=chunk)
                # time.sleep(0.001)
                state.frame_count += 1

                if state.frame_count % 200 == 0:
                    mode = "BYPASS" if bypass_ai else "AI"
                    logger.info(f"[{mode}] Frame {state.frame_count} | Errors: {state.error_count}")

            except (IOError, OSError) as e:
                state.error_count += 1
                err_msg = "Device disconnected or unavailable. Processing paused. Refresh devices and apply to resume."
                logger.error(f"Device error: {str(e)}")
                state.message_queue.put({'message': err_msg, 'isError': True, 'bypass_ai': bypass_ai})
                stop_event.set()
                break
            except Exception as e:
                state.error_count += 1
                err_msg = f"Processing error: {str(e)}. Check logs for details."
                logger.error(f"Frame processing error: {str(e)}")
                traceback.print_exc(file=open('audio_enhancer.log', 'a'))
                state.message_queue.put({'message': err_msg, 'isError': True, 'bypass_ai': bypass_ai})
                if state.error_count > max_errors:
                    logger.error("Too many processing errors, stopping...")
                    stop_event.set()
                    break
            
    except Exception as e:
        err_msg = f"Error starting audio streams: {str(e)}. Select different devices."
        logger.error(err_msg)
        state.message_queue.put({'message': err_msg, 'isError': True, 'bypass_ai': bypass_ai})
        stop_event.set()
    finally:
        try:
            if input_stream and input_stream.is_active():
                input_stream.stop_stream()
                input_stream.close()
            if output_stream and output_stream.is_active():
                output_stream.stop_stream()
                output_stream.close()
        except Exception as e:
            logger.error(f"Error during stream cleanup: {str(e)}")
        with state.lock:
            state.status = "Paused"
            state.message_queue.put({'message': 'Paused', 'isError': False, 'bypass_ai': bypass_ai})

# ---------------------------
# PyWebView API
# ---------------------------
class WebApi:
    def get_devices(self):
        refresh_devices()
        return {
            'inputs': [{'index': idx, 'name': name} for idx, name in state.inputs_list],
            'outputs': [{'index': idx, 'name': name} for idx, name in state.outputs_list]
        }

    def apply_changes(self, input_name, output_name, gate_open):
        global audio_thread_obj, stop_event
        new_input_index = None
        new_output_index = None
        for idx, name in state.inputs_list:
            if name == input_name:
                new_input_index = idx
                break
        for idx, name in state.outputs_list:
            if name == output_name:
                new_output_index = idx
                break
        if new_input_index is None or new_output_index is None:
            err_msg = "Invalid device selected. Refresh and try again."
            logger.error(err_msg)
            state.message_queue.put({'message': err_msg, 'isError': True, 'bypass_ai': state.bypass_ai})
            raise Exception(err_msg)

        with state.lock:
            state.gate_open = gate_open

        if audio_thread_obj and audio_thread_obj.is_alive():
            stop_event.set()
            audio_thread_obj.join(timeout=5.0)
            if audio_thread_obj.is_alive():
                logger.warning("Audio thread did not stop in time.")

        try:
            sample_rate, chunk, input_channels, output_channels = configure_audio_params(new_input_index, new_output_index)
            with state.lock:
                state.input_index = new_input_index
                state.output_index = new_output_index
                state.sample_rate = sample_rate
                state.chunk = chunk
                state.input_channels = input_channels
                state.output_channels = output_channels
                state.status = "Starting audio..."

            stop_event = threading.Event()
            audio_thread_obj = threading.Thread(target=audio_thread, args=(state, stop_event, model, df_state, pa))
            audio_thread_obj.start()
            logger.info("Applied changes and started new audio thread.")
            state.message_queue.put({'message': 'Changes applied. Audio processing started.', 'isError': False, 'bypass_ai': state.bypass_ai})
        except Exception as e:
            err_msg = f"Failed to apply changes: {str(e)}. Try different devices."
            logger.error(err_msg)
            state.message_queue.put({'message': err_msg, 'isError': True, 'bypass_ai': state.bypass_ai})
            raise

    def quit_app(self):
        global audio_thread_obj, stop_event
        if audio_thread_obj and audio_thread_obj.is_alive():
            stop_event.set()
            audio_thread_obj.join(timeout=5.0)
        pa.terminate()
        logger.info("Application quit.")
        with state.lock:
            while not state.message_queue.empty():
                state.message_queue.get()
        window.destroy()
        sys.exit(0)

    def get_status(self):
        with state.lock:
            status = state.status
            is_error = "Error" in status or "Paused" in status or "disconnected" in status
            bypass_ai = state.bypass_ai
        return {'message': status, 'isError': is_error, 'bypass_ai': bypass_ai}

    def get_pending_status(self):
        with state.lock:
            if state.message_queue.empty():
                return {'message': None, 'isError': False, 'bypass_ai': state.bypass_ai}
            messages = []
            while not state.message_queue.empty() and len(messages) < 10:
                messages.append(state.message_queue.get())
            return messages[-1] if messages else {'message': None, 'isError': False, 'bypass_ai': state.bypass_ai}
# ---------------------------
# Main Application
# ---------------------------
def start_webview():
    global window
    window = webview.create_window(
        'Noise Cancellation Control',
        html=HTML_CONTENT,
        js_api=WebApi(),
        width=700,
        height=500,
        resizable=False
    )
    webview.start()

if __name__ == '__main__':
    refresh_devices()
    try:
        default_input = pa.get_default_input_device_info()['index']
        default_output = pa.get_default_output_device_info()['index']
        for idx, name in state.inputs_list:
            if idx == default_input:
                state.input_index = idx
                break
        for idx, name in state.outputs_list:
            if idx == default_output:
                state.output_index = idx
                break
    except Exception:
        logger.warning("Could not set default devices.")
    if BYPASS_AI_PROCESSING:
        state.message_queue.put("Bypass AI mode enabled.")
    start_webview()
