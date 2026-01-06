# NOVA

**Lowlight Speed Detection Project**

NOVA is a computer vision project focused on speed detection in low-light conditions. The project is currently under active development.

## Overview

NOVA aims to detect and measure the speed of objects in challenging low-light environments using RTSP camera feeds and computer vision techniques. This project is designed to work with IP cameras (specifically tested with Tapo C500) to capture and process video streams for speed detection.

## Project Status

🚧 **Currently in Development** - This project is actively being developed and is not yet production-ready.

## Features

- RTSP stream connection and management
- Low-latency video streaming with buffer optimization
- Automatic reconnection handling for dropped connections
- Frame processing and display
- Speed detection capabilities (in development)

## Setup

### Prerequisites

- Python 3.11 or higher
- OpenCV (cv2) with FFmpeg support
- NumPy
- Network access to RTSP camera

### Installation

1. Clone or download this repository

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/Mac:**
     ```bash
     source venv/bin/activate
     ```

4. Install required dependencies:
   ```bash
   pip install opencv-python numpy
   ```

## Configuration

### RTSP Camera Setup (`rstp_cam.py`)

The `rstp_cam.py` script is used for testing and setting up the RTSP camera connection. Before running, you'll need to configure the following parameters in the script:

- **Username**: Your camera username (currently set to `@clarence123`)
- **Password**: Your camera password (currently set to `@clarence123`)
- **IP Address**: Your camera's IP address (currently set to `192.168.0.93`)

Edit these values in `rstp_cam.py`:
```python
username = quote("@clarence123")
password = quote("@clarence123")
ip = "192.168.0.93"
```

## Usage

### Testing RTSP Camera Connection

Run the camera setup script to test your RTSP connection:

```bash
python rstp_cam.py
```

This script will:
- Attempt to connect to the RTSP stream
- Try alternative stream paths if the default fails (`/stream1`, `/stream2`, `/h264`, `/live`)
- Display the video feed in a window
- Automatically reconnect if the connection drops
- Press 'q' to quit

### Features of the Test Script

- **Low-latency streaming**: Buffer size set to 1 for minimal delay
- **Automatic reconnection**: Handles dropped frames and reconnects automatically
- **Multiple stream path support**: Tries various common RTSP paths
- **Frame drop monitoring**: Tracks consecutive failures and reconnects when needed

## Requirements

- `opencv-python` - For video capture and processing
- `numpy` - For numerical operations

## Development Notes

- The project is currently in active development
- `rstp_cam.py` serves as a setup/testing utility for camera connectivity
- Speed detection algorithms are being developed
- Low-light optimization is a key focus area

## Troubleshooting

### Connection Issues

If you're having trouble connecting to the RTSP stream:

1. Verify your camera's IP address is correct
2. Ensure your camera supports RTSP streaming
3. Check that your username and password are correct
4. Verify network connectivity to the camera
5. The script will automatically try alternative stream paths

### Frame Drops

If you experience frequent frame drops:
- Check your network connection quality
- Ensure the camera is not overloaded
- Verify the camera's RTSP settings are configured correctly

## License

[Add your license information here]

## Contributing

This project is currently in active development. Contributions and feedback are welcome as the project evolves.

