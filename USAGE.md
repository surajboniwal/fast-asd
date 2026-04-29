# fast-asd Usage

Active speaker detection for local video files. Returns per-frame face positions and whether each face is speaking.

## Installation

```bash
# In your project's venv
uv pip install /path/to/fast-asd

# Also requires ffmpeg
brew install ffmpeg
```

## Setup

Initialize the models once at startup — this is the expensive step (loads ~400MB of weights into memory).

```python
from fast_asd.talknet import demoTalkNet
from fast_asd.talknet.talkNet import talkNet

s = talkNet()
s.loadParameters(demoTalkNet.pretrained_model_path)
s.eval()

DET = demoTalkNet.initialize_detector(device='cpu')  # use 'cuda' if GPU available
```

Model weights are auto-downloaded on first use to `~/.cache/fast-asd/`.

## Running on a video

```python
results = demoTalkNet.main(
    s,
    DET,
    video_path='myvideo.mp4',
    start_seconds=0,       # where to start in the video
    end_seconds=30,        # where to stop (omit or pass None for full video)
    return_visualization=False,  # True writes an annotated video to save/pyavi/video_out.mp4
)
```

## Return value

A list of frame objects, one per frame at the **original video's FPS** between `start_seconds` and `end_seconds`.

```python
[
  {
    'frame_number': 42,       # frame index in the original video
    'faces': [                # list of all detected faces in this frame
      {
        'track_id': 0,        # stable ID across frames (same person = same ID within a scene)
        'speaking': True,     # True if this face is the active speaker
        'raw_score': 0.6,     # confidence: positive = speaking, negative = silent
        'x1': 120,            # bounding box (pixels in original video resolution)
        'y1': 80,
        'x2': 220,
        'y2': 200,
      },
      {
        'track_id': 1,
        'speaking': False,
        'raw_score': -0.3,
        'x1': 400, 'y1': 90, 'x2': 500, 'y2': 210,
      }
    ]
  },
  ...
]
```

## Common queries

```python
# All speaking faces in a specific frame
frame = results[42]
speaking = [f for f in frame['faces'] if f['speaking']]

# Find all frames where person with track_id=0 is speaking
person_0_speaking = [
    r['frame_number']
    for r in results
    for f in r['faces']
    if f['track_id'] == 0 and f['speaking']
]

# Frames with no detected faces
empty_frames = [r['frame_number'] for r in results if not r['faces']]
```

## Notes

- **`track_id` resets per scene.** If the video cuts to a new scene, IDs start from 0 again. Don't use track_id to follow a person across cuts.
- **Minimum track length is 10 frames.** Faces that appear for fewer than 10 consecutive frames are discarded.
- **A `save/` directory** is written to your working directory on every run (extracted audio, frames, intermediate results). It is wiped and recreated each run.
- **`raw_score` threshold is 0.** Scores above 0 = speaking, below 0 = silent. The magnitude indicates confidence.
- Processing is CPU-only by default. On a Mac, expect ~2-5x slower than real-time depending on video resolution and number of faces.
