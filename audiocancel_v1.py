import pyaudio
import numpy as np
import torch
from df.enhance import enhance, init_df
from df import config
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

BYPASS_AI_PROCESSING = False 


def handle_channel_conversion(audio_np, input_channels, output_channels):
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
        print(f"⚠️  Unexpected channel config: {input_channels}→{output_channels}")
        return audio_np

def process_with_ai(audio_np, input_channels, output_channels, model, df_state):
    if input_channels == 1:
        mono_audio = audio_np
    elif input_channels == 2:
        stereo_audio = audio_np.reshape(-1, 2)
        mono_audio = np.mean(stereo_audio, axis=1)
    else:
        mono_audio = audio_np
    
    audio_tensor = torch.from_numpy(mono_audio).unsqueeze(0)
    
    with torch.no_grad():
        enhanced_tensor = enhance(model, df_state, audio_tensor, atten_lim_db=20.0)
    
    enhanced_mono = enhanced_tensor.squeeze(0).detach().cpu().numpy()
    
    
    if output_channels == 1:
        return enhanced_mono
    elif output_channels == 2:
        
        return np.column_stack([enhanced_mono, enhanced_mono])
    else:
        
        return enhanced_mono
pa = pyaudio.PyAudio()

def filter_common_devices(devices, device_type):
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

print("\n" + "="*60)
print("📱 MAIN INPUT DEVICES (Microphones):")
print("="*60)
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
    
    if 'microphone' in device_name.lower() or 'mic' in device_name.lower():
        emoji = "🎤"
    elif 'headset' in device_name.lower():
        emoji = "🎧"
    elif 'cable' in device_name.lower():
        emoji = "🔌"
    else:
        emoji = "📡"
        
    print(f"{emoji} {i}: {device_name} ({channels} channels, {sample_rate} Hz)")

print("\n" + "="*60)
print("🔊 MAIN OUTPUT DEVICES (Speakers/Virtual Cables):")
print("="*60)
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
    
    if 'speaker' in device_name.lower():
        emoji = "🔊"
    elif 'headphone' in device_name.lower() or 'headset' in device_name.lower():
        emoji = "🎧"
    elif 'cable' in device_name.lower():
        emoji = "🔌"
    else:
        emoji = "📡"
        
    print(f"{emoji} {i}: {device_name} ({channels} channels, {sample_rate} Hz)")

print(f"\n💡 Showing {len(filtered_inputs)} input and {len(filtered_outputs)} output devices")
print("🔍 Need to see ALL devices? Set show_all_devices = True in the code")

if BYPASS_AI_PROCESSING:
    print("\n🚨 DEBUG MODE: AI PROCESSING IS BYPASSED - Testing raw audio passthrough")
else:
    print("\n✅ AI PROCESSING ENABLED")

input_device_index = int(input("\n🎤 Enter INPUT device index: "))
output_device_index = int(input("🔊 Enter OUTPUT device index: "))

input_info = pa.get_device_info_by_index(input_device_index)
output_info = pa.get_device_info_by_index(output_device_index)

input_sample_rate = int(input_info['defaultSampleRate'])
output_sample_rate = int(output_info['defaultSampleRate'])

if input_sample_rate != output_sample_rate:
    print(f"⚠️  Sample rate mismatch: Input={input_sample_rate}Hz, Output={output_sample_rate}Hz")
    common_rates = [44100, 48000, 22050, 16000]
    SAMPLE_RATE = None
    for rate in common_rates:
        if rate <= min(input_sample_rate, output_sample_rate):
            SAMPLE_RATE = rate
            break
    if SAMPLE_RATE is None:
        SAMPLE_RATE = min(input_sample_rate, output_sample_rate)
    print(f"🔧 Using compromise sample rate: {SAMPLE_RATE}Hz")
else:
    SAMPLE_RATE = input_sample_rate
    print(f"✅ Sample rates match: {SAMPLE_RATE}Hz")

latency_target = 0.128
CHUNK = int(SAMPLE_RATE * latency_target)

CHUNK = ((CHUNK + 63) // 64) * 64
actual_latency = CHUNK / SAMPLE_RATE

print(f"🎯 Target latency: {latency_target*1000:.1f}ms")
print(f"📏 Calculated chunk size: {CHUNK} samples ({actual_latency*1000:.1f}ms actual latency)")


INPUT_CHANNELS = min(2, input_info['maxInputChannels'])
OUTPUT_CHANNELS = min(2, output_info['maxOutputChannels'])

if input_info['maxInputChannels'] == 0:
    print("❌ Selected input device has no input channels!")
    exit(1)
if output_info['maxOutputChannels'] == 0:
    print("❌ Selected output device has no output channels!")
    exit(1)


INPUT_CHANNELS = max(1, INPUT_CHANNELS)
OUTPUT_CHANNELS = max(1, OUTPUT_CHANNELS)

FORMAT = pyaudio.paFloat32

print("\n" + "="*60)
print("🔍 DIAGNOSTIC INFORMATION:")
print("="*60)
print(f"Input Device: {input_info['name']}")
print(f"  Max Input Channels: {input_info['maxInputChannels']}")
print(f"  Default Sample Rate: {input_info['defaultSampleRate']} Hz")
print(f"  Default Low Input Latency: {input_info['defaultLowInputLatency']:.3f}s")

print(f"\nOutput Device: {output_info['name']}")
print(f"  Max Output Channels: {output_info['maxOutputChannels']}")
print(f"  Default Sample Rate: {output_info['defaultSampleRate']} Hz")
print(f"  Default Low Output Latency: {output_info['defaultLowOutputLatency']:.3f}s")

print(f"\n🎛️  FINAL AUDIO CONFIGURATION:")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")
print(f"  Chunk Size: {CHUNK} samples ({actual_latency*1000:.1f}ms latency)")
print(f"  Input Channels: {INPUT_CHANNELS} (device supports {input_info['maxInputChannels']})")
print(f"  Output Channels: {OUTPUT_CHANNELS} (device supports {output_info['maxOutputChannels']})")
print(f"  Audio Format: Float32")


print(f"\n📋 CHANNEL MAPPING STRATEGY:")
if INPUT_CHANNELS == 1 and OUTPUT_CHANNELS == 1:
    print("  Strategy: Mono → Mono (direct pass-through)")
elif INPUT_CHANNELS == 1 and OUTPUT_CHANNELS == 2:
    print("  Strategy: Mono → Stereo (duplicate mono to L+R channels)")
elif INPUT_CHANNELS == 2 and OUTPUT_CHANNELS == 1:
    print("  Strategy: Stereo → Mono (mix L+R channels)")
elif INPUT_CHANNELS == 2 and OUTPUT_CHANNELS == 2:
    print("  Strategy: Stereo → Stereo (direct pass-through)")


if input_info['maxInputChannels'] > 2:
    print(f"  ℹ️  Input device has {input_info['maxInputChannels']} channels but using only {INPUT_CHANNELS}")
if output_info['maxOutputChannels'] > 2:
    print(f"  ℹ️  Output device has {output_info['maxOutputChannels']} channels but using only {OUTPUT_CHANNELS}")
    
print("="*60)

if not BYPASS_AI_PROCESSING:
    print("\nLoading DeepFilterNet model...")
    try:

        model, df_state, _ = init_df()
        if torch.cuda.is_available():
            model=model.cuda()
            print("USING GPU POWER!!!!")
        print("Model loaded successfully!")
        
        print("Testing enhance function...")
        test_audio = np.random.randn(CHUNK).astype(np.float32)  
        test_tensor = torch.from_numpy(test_audio).unsqueeze(0)
        
        with torch.no_grad():
            enhanced_test = enhance(model, df_state, test_tensor, atten_lim_db=20.0)
        
        enhanced_np = enhanced_test.squeeze(0).detach().cpu().numpy()
        print(f"Test successful! Input shape: {test_tensor.shape}, Output shape: {enhanced_test.shape}")
        print(f"Output numpy shape: {enhanced_np.shape}")
        
    except Exception as e:
        print(f"Error during model loading or testing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
else:
    print("\n⏭️ Skipping model loading (DEBUG MODE)")

print(f"\nFinal Audio Settings Applied:")
print(f"  Sample Rate: {SAMPLE_RATE} Hz")
print(f"  Chunk Size: {CHUNK} samples ({CHUNK/SAMPLE_RATE*1000:.1f}ms)")
print(f"  Format: Float32")

try:
    buffer_multiplier = 4  
    
    input_stream = pa.open(format=FORMAT,
                          channels=INPUT_CHANNELS,
                          rate=SAMPLE_RATE,
                          input=True,
                          input_device_index=input_device_index,
                          frames_per_buffer=CHUNK,
                          stream_callback=None,  
                          start=False)

    output_stream = pa.open(format=FORMAT,
                           channels=OUTPUT_CHANNELS,
                           rate=SAMPLE_RATE,
                           output=True,
                           output_device_index=output_device_index,
                           frames_per_buffer=CHUNK,
                           stream_callback=None,  
                           start=False)
    
    
    input_stream.start_stream()
    import time
    time.sleep(0.01)  
    output_stream.start_stream()
    
    print(f"\n✅ Audio streams created with {CHUNK} sample buffers!")
    
except Exception as e:
    print(f"❌ Error creating audio streams: {e}")
    pa.terminate()
    exit(1)

if BYPASS_AI_PROCESSING:
    print(f"\n🎵 RAW AUDIO PASSTHROUGH active...")
else:
    print(f"\n🤖 AI NOISE SUPPRESSION active...")

print(f"Input: {input_info['name']}")
print(f"Output: {output_info['name']}")
print("Press Ctrl+C to stop.\n")

frame_count = 0
error_count = 0
max_errors = 10

print("🎵 Starting audio processing...")
print("📊 Monitoring for dropouts and buffer issues...")

try:
    
    import time
    time.sleep(0.1)
    
    while True:
        try:
            try:
                data = input_stream.read(CHUNK, exception_on_overflow=False)
            except Exception as read_error:
                print(f"📥 Input read error: {read_error}")
                error_count += 1
                if error_count > max_errors:
                    print("❌ Too many input errors, stopping...")
                    break
                continue
                
            audio_np = np.frombuffer(data, dtype=np.float32)
            
            
            expected_input_samples = CHUNK * INPUT_CHANNELS
            if len(audio_np) != expected_input_samples:
                print(f"⚠️  Input buffer size mismatch: expected {expected_input_samples}, got {len(audio_np)}")
            
                if len(audio_np) < expected_input_samples:
                    padding = np.zeros(expected_input_samples - len(audio_np), dtype=np.float32)
                    audio_np = np.concatenate([audio_np, padding])
                else:
                    audio_np = audio_np[:expected_input_samples]
            
    
            audio_level = np.abs(audio_np).mean()
            if frame_count > 10 and audio_level < 1e-6:  
                if frame_count % 100 == 0:  
                    print(f"🔇 Low audio level detected: {audio_level:.2e}")
            
            if BYPASS_AI_PROCESSING:
                
                enhanced = handle_channel_conversion(audio_np, INPUT_CHANNELS, OUTPUT_CHANNELS)
            else:

                try:
                    enhanced = process_with_ai(audio_np, INPUT_CHANNELS, OUTPUT_CHANNELS, model, df_state)
                except Exception as ai_error:
                    print(f"🤖 AI processing error: {ai_error}")

                    enhanced = handle_channel_conversion(audio_np, INPUT_CHANNELS, OUTPUT_CHANNELS)
            
            
            enhanced = np.clip(enhanced.astype(np.float32), -0.9, 0.9)  
            
            if frame_count < 5:  
                fade = frame_count / 5.0
                enhanced = enhanced * fade
            
            expected_output_samples = CHUNK * OUTPUT_CHANNELS
            if enhanced.size != expected_output_samples:
                print(f"⚠️  Output size mismatch: expected {expected_output_samples}, got {enhanced.size}")
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
                    enhanced = enhanced.flatten()[:expected_output_samples]
            
            try:
                output_data = np.ascontiguousarray(enhanced).tobytes()
                output_stream.write(output_data, num_frames=CHUNK)
            except Exception as write_error:
                print(f"📤 Output write error: {write_error}")
                error_count += 1
                if error_count > max_errors:
                    print("❌ Too many output errors, stopping...")
                    break
                continue
            
            frame_count += 1
            
            if frame_count % 200 == 0:
                mode = "BYPASS" if BYPASS_AI_PROCESSING else "AI"
                avg_level = np.abs(enhanced).mean()
                print(f"📊 [{mode}] Frame {frame_count} | Audio level: {avg_level:.4f} | Errors: {error_count}", end='\r')
                
        except KeyboardInterrupt:
            raise  
        except Exception as inner_e:
            print(f"⚠️  Frame processing error: {inner_e}")
            error_count += 1
            if error_count > max_errors:
                print("❌ Too many processing errors, stopping...")
                break
            continue
        
except KeyboardInterrupt:
    print(f"\n\nUser stopped after {frame_count} frames...")
except Exception as e:
    print(f"❌ Critical error: {e}")
    import traceback
    traceback.print_exc()
finally:
    try:
        if 'input_stream' in locals() and input_stream.is_active():
            input_stream.stop_stream()
            input_stream.close()
        if 'output_stream' in locals() and output_stream.is_active():
            output_stream.stop_stream()
            output_stream.close()
    except:
        pass
    pa.terminate()
    
    print(f"\n📊 Session Stats:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total errors: {error_count}")
    print(f"  Success rate: {((frame_count-error_count)/max(frame_count,1)*100):.1f}%")
    
    if BYPASS_AI_PROCESSING:
        print("\n💡 If bypass mode still has cuts/distortion:")
        print("   1. Try different USB ports for your headset")
        print("   2. Check Windows audio enhancements (disable them)")
        print("   3. Try different sample rates in Windows sound settings")
    else:
        print("\n💡 To test without AI processing, set BYPASS_AI_PROCESSING = True at the top.")