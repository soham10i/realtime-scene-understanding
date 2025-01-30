# Real-Time Scene Understanding

This project processes a video to detect objects, generate captions for them, and summarize the video content using YOLOv8 and BLIP models.

## Prerequisites

- Python 3.7+
- pip (Python package installer)

## Installation

1. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Set the paths for the model and video in the `main` section of `captions.py`:

   ```python
   model_path = "yolov8n-seg.pt"
   video_path = "classroom_video_01.mp4"
   ```
2. Run the script:

   ```sh
   python captions.py
   ```

## Output

- The processed video will be saved as `processed_video.mp4`.
- Captions for each frame will be logged in `captions_log.txt`.
- A summary of the video will be saved in `video_summary_led.txt`.

## Troubleshooting

- Ensure that the paths to the model and video are correct.
- Verify that all required packages are installed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
