### Adjusted README for GitHub (General Public Audience)

# Audio Enhancer

A lightweight, open-source audio enhancement tool designed to improve audio quality by reducing background noise. Perfect for calls, recordings, or streaming, this tool offers a user-friendly alternative to similar solutions like Krisp, with compatibility for virtual audio drivers such as VB Audio Virtual Cable.

## Overview
The Audio Enhancer leverages DeepFilterNet3 for noise suppression and provides a simple PyWebView-based interface. It’s designed to be packaged as a standalone EXE for Windows users and can be built from source for customization. This project is ideal for anyone looking to enhance audio quality without complex setups.

## Features
- Removes background noise (e.g., fan hum, keyboard clicks) for clearer voice output.
- Compatible with VB Audio Virtual Cable for audio routing to applications.
- Intuitive UI for selecting input/output devices and enabling noise gate.
- Built with Python and PyInstaller for easy distribution.

## Prerequisites
- **Windows OS** (tested on Windows 10/11).
- **Python 3.8+** (for development or building from source).
- **VB Audio Virtual Cable** (optional, download from [vb-audio.com/Cable/](https://vb-audio.com/Cable/)) for virtual audio routing.
- Required Python packages (see `requirements.txt`).

## Installation

### For Development
1. Clone the repository:
   ```bash
   git clone https://github.com/noor8271/EZ_Krisp.git
   cd AudioEnhancer
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure DeepFilterNet3 files (`model_120.ckpt.best`, `config.ini`) are in the `DeepFilterNet3` folder or adjust `resource_path` accordingly.


## Usage
AFTER BUILDING THE EXE.
1. **Run the EXE**:
   - Double-click `AudioEnhancer.exe`. It may take 15-20 seconds to load (normal behavior—please wait).
   - If Windows SmartScreen warns you, click "More info" then "Run anyway" to proceed.
2. **Configure Settings**:
   - In the window, select your microphone under **"Input Device"**.
   - Choose **"VB Audio Virtual Input"** (if installed) as the **"Output Device"**, or use your default output.
   - Leave **"Enable Noise Gate"** on (green) to reduce background noise when not speaking.
   - Click **"Apply Changes"** to start processing.
3. **Use Enhanced Audio**:
   - In your app (e.g., Zoom, Discord, OBS), set the microphone to **"VB Audio Virtual Output"** (if using VB Cable) or the output device selected.
   - Test by speaking—your audio should be clearer!
4. **Refresh Devices**:
   - If microphones or outputs don’t appear, click **"Refresh Devices"** to update the list, then reapply settings.
5. **Quit**:
   - Click **"Quit"** to close the program.

## Building the EXE
To create a standalone executable:
1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Build the EXE:
   ```bash
   pyinstaller --onefile --windowed --add-data "DeepFilterNet3;DeepFilterNet3" audio_enhancer.py
   ```
3. Find the EXE in the `dist` folder. Ensure DeepFilterNet3 files are included in the project directory.

## Known Issues
- Initial startup may take 15-20 seconds due to model loading.
- Windows SmartScreen may block the EXE on first run—users must allow it manually.
- Requires VB Audio Virtual Cable for full virtual audio functionality (optional).


## License
[MIT License](LICENSE) - Feel free to use, modify, and distribute this code, but please include the license and attribution.

## Feedback
If you encounter issues or have suggestions, shoot an email at noor2021@namal.edu.pk

## Acknowledgments
- Built using [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) for noise suppression.
- UI powered by [PyWebView](https://pywebview.flowrl.com/).
