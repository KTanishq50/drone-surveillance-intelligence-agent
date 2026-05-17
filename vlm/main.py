import os
import json
import uuid
import cv2
import torch

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText
)

# ---------------------------------------------------
# folders
# ---------------------------------------------------

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
FRAME_DIR = "temp_frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------------------------------------------
# fastapi
# ---------------------------------------------------

app = FastAPI()

# ---------------------------------------------------
# load model
# ---------------------------------------------------

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

print("loading processor...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

print("loading model...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32
)

print("model loaded")

# ---------------------------------------------------
# process video
# ---------------------------------------------------

def process_video(video_path):

    video = cv2.VideoCapture(video_path)

    frame_count = 0

    results = []

    while True:

        success, frame = video.read()

        if not success:
            break

        # process every 30 frames
        if frame_count % 30 == 0:

            print(f"processing frame {frame_count}")

            # convert frame
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(frame_rgb)

            # resize for faster inference
            image = image.resize((320, 240))

            prompt = (
                "You are a surveillance AI system. "
                "Describe this security camera frame "
                "in concise security-report style."
            )

            # prepare inputs
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )

            # inference
            output = model.generate(
                **inputs,
                max_new_tokens=40
            )

            # decode
            result = processor.batch_decode(
                output,
                skip_special_tokens=True
            )[0]

            print(result)

            results.append({
                "frame_id": len(results) + 1,
                "description": result
            })

        frame_count += 1

    video.release()

    return results

# ---------------------------------------------------
# upload route
# ---------------------------------------------------

@app.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...)
):

    unique_name = f"{uuid.uuid4()}.mp4"

    video_path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    # save uploaded file
    with open(video_path, "wb") as f:
        f.write(await file.read())

    print("video uploaded")

    # process video
    results = process_video(video_path)

    # output json path
    output_path = os.path.join(
        OUTPUT_DIR,
        "frames.json"
    )

    # save json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return JSONResponse({
        "status": "done",
        "json_file": output_path,
        "frames_processed": len(results)
    })