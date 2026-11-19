
import av
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_DIR = BASE_DIR / "evaluation" / "frames"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

container = av.open(str(VIDEO_PATH))
stream = container.streams.video[0]

fps = float(stream.average_rate)
frame_interval = max(1, round(fps))

saved_count = 0
total_frames = 0

print(f"Video FPS: {fps:.2f}")
print("Extracting frames...")

for frame_number, frame in enumerate(container.decode(stream)):
    total_frames += 1

    if frame_number % frame_interval == 0:

        image = frame.to_image()

        filename = f"frame_{frame_number:05d}.png"

        image.save(OUTPUT_DIR / filename)

        saved_count += 1

        print(f"Saved: {filename}")

container.close()

print("\n--- Extraction Complete ---")
print(f"Total video frames: {total_frames}")
print(f"Extracted images: {saved_count}")
print(f"Output folder: {OUTPUT_DIR}")
