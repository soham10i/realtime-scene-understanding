import os
import cv2
import torch
from PIL import Image
from ultralytics import YOLO  # YOLOv8 library
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM, LongformerConfig, LongformerModel, LongformerTokenizer
import json


class CaptionGenerator:
    def __init__(self, model_path: str):
        """
        Initializes the CaptionGenerator with the specified model path.

        Args:
            model_path (str): Path to the model.
        """
        self.model_path = model_path
        self.yolo_model = self.setup_yolov8()
        self.processor, self.blip_model = self.setup_blip()

    def setup_yolov8(self):
        """
        Sets up the YOLOv8 model for segmentation and detection.

        Returns:
            YOLO: The YOLOv8 model.
        """
        try:
            print("Loading YOLOv8 model...")
            model = YOLO('yolov8n-seg.pt')
            if torch.cuda.is_available():
                model.to('cuda')
                print("YOLOv8 model loaded successfully on GPU.")
            else:
                print("YOLOv8 model loaded successfully on CPU.")
            return model
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            raise

    def setup_blip(self):
        """
        Sets up the BLIP model and processor.

        Returns:
            tuple: The BLIP processor and model.
        """
        try:
            print("Loading BLIP model and processor...")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            if torch.cuda.is_available():
                model.to('cuda')
                print("BLIP model loaded successfully on GPU.")
            else:
                print("BLIP model loaded successfully on CPU.")
            return processor, model
        except Exception as e:
            print(f"Error loading BLIP model and processor: {e}")
            raise

    def generate_caption(self, image):
        """
        Generates a caption for a given image using the BLIP model.

        Args:
            image (PIL.Image.Image): The image to generate a caption for.

        Returns:
            str: The generated caption.
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
            output = self.blip_model.generate(**inputs, max_new_tokens=128)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Error generating caption"

    def save_captions(self, captions, output_path):
        """
        Saves the generated captions to a JSON file.

        Args:
            captions (list): List of generated captions.
            output_path (str): Path to save the captions JSON file.
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(captions, f)
        except Exception as e:
            print(f"Error saving captions: {e}")


class VideoProcessor:
    def __init__(self, video_path: str, caption_generator: CaptionGenerator):
        """
        Initializes the VideoProcessor with the specified video path and caption generator.

        Args:
            video_path (str): Path to the video file.
            caption_generator (CaptionGenerator): Instance of CaptionGenerator.
        """
        self.video_path = video_path
        self.caption_generator = caption_generator
        self.captions_file = "captions_log.txt"
        self.summary_file = "video_summary_led.txt"

    def draw_bounding_boxes_and_labels(self, frame, results, captions):
        """
        Draws bounding boxes, class labels, and captions on the frame.

        Args:
            frame (numpy.ndarray): The frame to annotate.
            results (Results): YOLOv8 detection results.
            captions (list): Captions generated for detected objects.

        Returns:
            numpy.ndarray: Annotated frame.
        """
        try:
            for idx, (box, cls, caption) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls, captions)):
                x1, y1, x2, y2 = map(int, box.tolist())
                class_name = results[0].names[int(cls)]
                label = f"{class_name}: {caption}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            return frame
        except Exception as e:
            print(f"Error drawing bounding boxes and labels: {e}")
            return frame

    def process_yolo_and_generate_captions(self, results, frame):
        """
        Processes YOLOv8 results and generates captions for detected objects.

        Args:
            results (Results): YOLOv8 detection results.
            frame (numpy.ndarray): The current video frame.

        Returns:
            list: Captions for each detected object.
        """
        captions = []
        try:
            frame_height, frame_width = frame.shape[:2]

            for box, mask in zip(results[0].boxes.xyxy, results[0].masks.data):
                x1, y1, x2, y2 = map(int, box.tolist())

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)

                mask = (mask.cpu().numpy() * 255).astype("uint8")
                roi = frame[y1:y2, x1:x2]

                if roi.size > 0:
                    roi_height, roi_width = roi.shape[:2]
                    resized_mask = cv2.resize(mask, (roi_width, roi_height), interpolation=cv2.INTER_NEAREST)

                    if resized_mask.shape[:2] == roi.shape[:2]:
                        masked_roi = cv2.bitwise_and(roi, roi, mask=resized_mask)
                        pil_image = Image.fromarray(cv2.cvtColor(masked_roi, cv2.COLOR_BGR2RGB))
                        captions.append(self.caption_generator.generate_caption(pil_image))
                    else:
                        captions.append("Invalid mask and ROI size mismatch")
                else:
                    captions.append("No valid ROI detected")

        except Exception as e:
            print(f"Error processing YOLO results and generating captions: {e}")
            captions.append("Error processing YOLO results")

        return captions

    def summarize_with_longformer(self):
        """
        Summarizes the content of the video using captions from the captions log file, leveraging Longformer.

        Returns:
            str: Generated summary of the video.
        """
        try:
            with open(self.captions_file, "r") as f:
                captions_content = f.readlines()

            captions = []
            for line in captions_content:
                if "Object" in line:
                    caption = line.split(": ", 1)[-1].strip()
                    if caption not in captions:
                        captions.append(caption)

            captions_text = " ".join(captions)

            model_name = "allenai/longformer-base-4096"
            tokenizer = LongformerTokenizer.from_pretrained(model_name)
            model = LongformerModel.from_pretrained(model_name)

            inputs = tokenizer(captions_text, return_tensors="pt", max_length=4096, truncation=True)

            summary_ids = model.generate(inputs.input_ids, max_length=500, min_length=100, length_penalty=2.0, num_beams=4)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            with open(self.summary_file, "w") as f:
                f.write(summary)

            print(f"Summary saved to: {self.summary_file}")
            return summary
        except Exception as e:
            print(f"Error summarizing video: {e}")
            return "Error summarizing video"

    def process_video(self):
        """
        Processes the video, performs YOLOv8 detection, generates captions, and summarizes the video content.
        """
        try:
            open(self.captions_file, "w").close()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                return

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter("processed_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

            print("Processing video...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.caption_generator.yolo_model(frame, conf=0.5, iou=0.4)
                captions = self.process_yolo_and_generate_captions(results, frame)

                with open(self.captions_file, "a") as f:
                    f.write(f"Frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}:\n")
                    for idx, caption in enumerate(captions):
                        f.write(f"  Object {idx + 1}: {caption}\n")
                    f.write("\n")

            cap.release()
            out.release()

            print("Generating video summary...")
            summary = self.summarize_with_longformer()
            print("Video Summary:")
            print(summary)
        except Exception as e:
            print(f"Error processing video: {e}")


if __name__ == "__main__":
    try:
        model_path = "path/to/model"
        video_path = "classroom_video_01.mp4"

        caption_generator = CaptionGenerator(model_path)
        video_processor = VideoProcessor(video_path, caption_generator)
        video_processor.process_video()
    except Exception as e:
        print(f"Error in main execution: {e}")
