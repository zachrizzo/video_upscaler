from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

# Load the model for image upscaling
model_path = 'models/RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

def upscale_frame(frame):
    # Implement the upscaling of a single frame using the RRDBNet model
    # Similar to the image upscaling code you provided
    # ...

@app.post("/upscale-video/")
async def upscale_video(file: UploadFile = File(...)):
    contents = await file.read()
    video_stream = io.BytesIO(contents)
    cap = cv2.VideoCapture(video_stream)

    # Define codec and create VideoWriter object to write the upscaled video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('upscaled_video.mp4', fourcc, 20.0, (1920, 1080))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Upscale the frame
        upscaled_frame = upscale_frame(frame)

        # Write the frame
        out.write(upscaled_frame)

    cap.release()
    out.release()

    # Return the upscaled video
    file_like = open('upscaled_video.mp4', mode="rb")
    return StreamingResponse(file_like, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
