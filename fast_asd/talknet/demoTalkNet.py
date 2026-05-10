import glob
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from shutil import rmtree

import cv2
import numpy
import python_speech_features
import torch
import tqdm
from scenedetect.detectors import ContentDetector
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile

from .model.faceDetector.s3fd import S3FD
from .talkNet import talkNet

warnings.filterwarnings("ignore")

pretrained_model_path = os.path.join(
    os.path.expanduser("~"), ".cache", "fast-asd", "pretrain_TalkSet.model"
)
save_path = "/tmp/save/"
data_loader_thread = 10
face_detection_scale = 0.25
min_track = 10  # Number of min frames for each shot
num_failed_det = 10  # Number of missed detections allowed before tracking is stopped
min_face_size = 1  # Minimum face size in pixels
crop_scale = 0.40  # Scale bounding box
start = 0  # The start time of the video
duration = 0  # The duration of the video, when set as 0, will extract the whole video
pyaviPath = os.path.join(save_path, "pyavi")
pyframesPath = os.path.join(save_path, "pyframes")
pyworkPath = os.path.join(save_path, "pywork")
pycropPath = os.path.join(save_path, "pycrop")
videoFilePath = os.path.join(pyaviPath, "video.avi")
audioFilePath = os.path.join(pyaviPath, "audio.wav")

if os.path.isfile(pretrained_model_path) == False:  # Download the pretrained model
    os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
    Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
    cmd = "gdown %s -O %s" % (Link, pretrained_model_path)
    subprocess.call(cmd, shell=True, stdout=None)


def scene_detect(video_path, save=False, start_frame=0, end_frame=None):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([video_path])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    if start_frame != 0 or end_frame is not None:
        start_timecode = FrameTimecode(
            timecode=start_frame, fps=videoManager.get_framerate()
        )
        end_timecode = FrameTimecode(
            timecode=end_frame, fps=videoManager.get_framerate()
        )
        videoManager.set_duration(start_time=start_timecode, end_time=end_timecode)
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    if sceneList == []:
        if start_frame == 0 and end_frame is None:
            sceneList = [
                (videoManager.get_base_timecode(), videoManager.get_current_timecode())
            ]
        else:
            sceneList = [(start_timecode, end_timecode)]
    if save:
        savePath = os.path.join(pyworkPath, "scene.pckl")
        with open(savePath, "wb") as fil:
            pickle.dump(sceneList, fil)
            # sys.stderr.write('%s - scenes detected %d\n'%(video_path, len(sceneList)))
    return sceneList


def initialize_detector(device="cuda"):
    # Initialize the face detector
    DET = S3FD(device=device)
    return DET


_FACE_DETECT_BATCH = 64


def _build_dets(batch_results, batch_start_fidx, start_frame):
    dets = []
    for fidx, bboxes in enumerate(batch_results):
        global_fidx = batch_start_fidx + fidx
        dets.append([
            {
                "frame": global_fidx + start_frame,
                "bbox": (bbox[:-1]).tolist(),
                "conf": bbox[-1],
            }
            for bbox in bboxes
        ])
    return dets


def predict_faces(DET, start_frame=0):
    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
    flist.sort()
    dets = []
    for batch_start in range(0, len(flist), _FACE_DETECT_BATCH):
        batch_files = flist[batch_start : batch_start + _FACE_DETECT_BATCH]
        images = [
            cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in batch_files
        ]
        batch_results = DET.detect_faces_batch(
            images, conf_th=0.9, scales=[face_detection_scale]
        )
        dets.extend(_build_dets(batch_results, batch_start, start_frame))

    savePath = os.path.join(pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


def predict_faces_from_frames(DET, frames, start_frame=0):
    dets = []
    for batch_start in range(0, len(frames), _FACE_DETECT_BATCH):
        batch_frames = frames[batch_start : batch_start + _FACE_DETECT_BATCH]
        images = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
        batch_results = DET.detect_faces_batch(
            images, conf_th=0.9, scales=[face_detection_scale]
        )
        dets.extend(_build_dets(batch_results, batch_start, start_frame))

    savePath = os.path.join(pyworkPath, "faces.pckl")
    with open(savePath, "wb") as fil:
        pickle.dump(dets, fil)
    return dets


_FACE_DETECT_STEP = 5  # detect every 5th 25fps frame (= 5fps); track_shot interpolates gaps


def predict_faces_streaming(DET, video_path, fps, duration, start_seconds=0):
    """Run face detection by streaming frames from video in batches, no JPEG I/O.

    A producer thread decodes and preprocesses frames while the main thread runs
    GPU inference, overlapping I/O with compute.

    Face detection runs at _FACE_DETECT_STEP-frame intervals (5fps by default).
    track_shot's interp1d fills bbox positions for skipped frames; num_failed_det=10
    tolerates gaps up to 10 frames.
    """
    import queue
    import threading

    start_frame = int(start_seconds * fps)
    end_frame = int((start_seconds + duration) * fps)
    num_25fps_frames = int(duration * 25) + 1

    # (fps25_idx, orig_frame_idx) pairs — only every _FACE_DETECT_STEP-th 25fps frame
    sample_frames = []
    for i in range(0, num_25fps_frames, _FACE_DETECT_STEP):
        target_orig = start_frame + int(i * (fps / 25))
        if target_orig < end_frame:
            sample_frames.append((i, target_orig))

    def _producer(cap, q):
        batch_imgs = []
        batch_idxs = []
        current_orig = start_frame - 1
        current_image = None
        for fps25_idx, target_orig in sample_frames:
            while current_orig < target_orig:
                ret, frame = cap.read()
                if not ret:
                    if batch_imgs:
                        q.put((batch_idxs, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_imgs]))
                    cap.release()
                    q.put(None)
                    return
                current_orig += 1
                current_image = frame
            batch_imgs.append(current_image.copy())
            batch_idxs.append(fps25_idx)
            if len(batch_imgs) == _FACE_DETECT_BATCH:
                q.put((batch_idxs, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_imgs]))
                batch_imgs = []
                batch_idxs = []
        if batch_imgs:
            q.put((batch_idxs, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_imgs]))
        cap.release()
        q.put(None)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    q = queue.Queue(maxsize=2)
    producer = threading.Thread(target=_producer, args=(cap, q), daemon=True)
    producer.start()

    detected = {}
    while True:
        item = q.get()
        if item is None:
            break
        batch_idxs, images = item
        results = DET.detect_faces_batch(images, conf_th=0.9, scales=[face_detection_scale])
        for fps25_idx, bboxes in zip(batch_idxs, results):
            detected[fps25_idx] = [
                {"frame": fps25_idx, "bbox": bbox[:-1].tolist(), "conf": float(bbox[-1])}
                for bbox in bboxes
            ]

    producer.join()
    # Return one entry per 25fps frame; skipped frames are empty (track_shot handles gaps)
    return [detected.get(i, []) for i in range(num_25fps_frames)]


def crop_video_from_video(track, cropFile, video_path, fps, audio_path):
    """Crop face track by reading needed frames sequentially from the video file."""
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    orig_frames = [int(f * (fps / 25)) for f in track["frame"]]
    first_orig = orig_frames[0]
    last_orig = orig_frames[-1]

    # Map each original-fps frame number to the 25fps track indices that need it
    orig_to_idxs = {}
    for fidx, orig_f in enumerate(orig_frames):
        orig_to_idxs.setdefault(orig_f, []).append(fidx)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_orig)
    current_orig = first_orig

    while current_orig <= last_orig:
        ret, image = cap.read()
        if not ret:
            break
        if current_orig in orig_to_idxs:
            for fidx in orig_to_idxs[current_orig]:
                cs = crop_scale
                bs = dets["s"][fidx]
                bsi = int(bs * (1 + 2 * cs))
                padded = numpy.pad(
                    image,
                    ((bsi, bsi), (bsi, bsi), (0, 0)),
                    "constant",
                    constant_values=(110, 110),
                )
                my = dets["y"][fidx] + bsi
                mx = dets["x"][fidx] + bsi
                face = padded[
                    int(my - bs) : int(my + bs * (1 + 2 * cs)),
                    int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
                ]
                vOut.write(cv2.resize(face, (224, 224)))
        current_orig += 1

    cap.release()
    vOut.release()

    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (audio_path, data_loader_thread, audioStart, audioEnd, audioTmp)
    )
    subprocess.call(command, shell=True, stdout=None)
    _, audio = wavfile.read(audioTmp)
    command = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a aac %s.avi -loglevel panic"
        % (cropFile, audioTmp, data_loader_thread, cropFile)
    )
    subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}


def crop_video_from_video_all_tracks(tracks, pycropPath, video_path, fps, audio_path):
    """Crop all face tracks in a single sequential pass through the video.

    Reduces video I/O from N_tracks passes down to one pass per clip.
    """
    if not tracks:
        return []

    proc_tracks = []
    for track in tracks:
        dets = {"x": [], "y": [], "s": []}
        for det in track["bbox"]:
            dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets["y"].append((det[1] + det[3]) / 2)
            dets["x"].append((det[0] + det[2]) / 2)
        dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
        dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
        dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
        proc_tracks.append(dets)

    # Build orig-fps frame → [(track_idx, fidx)] mapping
    orig_to_entries = {}
    for tidx, track in enumerate(tracks):
        for fidx, frame_25fps in enumerate(track["frame"].tolist()):
            orig_f = int(frame_25fps * (fps / 25))
            orig_to_entries.setdefault(orig_f, []).append((tidx, fidx))

    if not orig_to_entries:
        return [{"track": t, "proc_track": p} for t, p in zip(tracks, proc_tracks)]

    first_orig = min(orig_to_entries)
    last_orig = max(orig_to_entries)

    vOuts = [
        cv2.VideoWriter(
            os.path.join(pycropPath, "%05d" % ii) + "t.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            25,
            (224, 224),
        )
        for ii in range(len(tracks))
    ]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_orig)
    current_orig = first_orig
    while current_orig <= last_orig:
        ret, image = cap.read()
        if not ret:
            break
        if current_orig in orig_to_entries:
            for tidx, fidx in orig_to_entries[current_orig]:
                dets = proc_tracks[tidx]
                cs = crop_scale
                bs = dets["s"][fidx]
                bsi = int(bs * (1 + 2 * cs))
                padded = numpy.pad(
                    image,
                    ((bsi, bsi), (bsi, bsi), (0, 0)),
                    "constant",
                    constant_values=(110, 110),
                )
                my = dets["y"][fidx] + bsi
                mx = dets["x"][fidx] + bsi
                face = padded[
                    int(my - bs) : int(my + bs * (1 + 2 * cs)),
                    int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
                ]
                vOuts[tidx].write(cv2.resize(face, (224, 224)))
        current_orig += 1
    cap.release()
    for vOut in vOuts:
        vOut.release()

    vid_tracks = []
    for ii, track in enumerate(tracks):
        cropFile = os.path.join(pycropPath, "%05d" % ii)
        audioTmp = cropFile + ".wav"
        audioStart = float(track["frame"][0]) / 25
        audioEnd = float(track["frame"][-1] + 1) / 25
        subprocess.call(
            "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
            % (audio_path, data_loader_thread, audioStart, audioEnd, audioTmp),
            shell=True,
        )
        subprocess.call(
            "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a aac %s.avi -loglevel panic"
            % (cropFile, audioTmp, data_loader_thread, cropFile),
            shell=True,
        )
        os.remove(cropFile + "t.avi")
        vid_tracks.append({"track": track, "proc_track": proc_tracks[ii]})
    return vid_tracks


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(sceneFaces):
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > min_track:
            frameNum = numpy.array([f["frame"] for f in track])
            bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = numpy.stack(bboxesI, axis=1)
            if (
                max(
                    numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                    numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                )
                > min_face_size
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks


def crop_video(track, cropFile, start_frame=0):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = crop_scale  # Crop scale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        # print(fidx, frame, len(flist))
        image = cv2.imread(flist[frame])
        frame = numpy.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (audioFilePath, data_loader_thread, audioStart, audioEnd, audioTmp)
    )
    output = subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a aac %s.avi -loglevel panic"
        % (cropFile, audioTmp, data_loader_thread, cropFile)
    )  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}


def crop_video_from_frames(track, cropFile, frames, start_frame=0, audio_path=None):
    # CPU: crop the face clips
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = crop_scale  # Crop scale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        # print(fidx, frame, len(flist))
        # print(fidx, frame, len(frames))
        image = frames[frame]
        frame = numpy.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()
    command = (
        "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic"
        % (audioFilePath, data_loader_thread, audioStart, audioEnd, audioTmp)
    )
    output = subprocess.call(command, shell=True, stdout=None)  # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = (
        "ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a aac %s.avi -loglevel panic"
        % (cropFile, audioTmp, data_loader_thread, cropFile)
    )  # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split("/")[-1].replace(".wav", ".npy"))
    numpy.save(featuresPath, mfcc)


def evaluate_network(s, files):
    # GPU: active speaker detection by pretrained TalkNet
    allScores = []
    durationSet = {1, 2, 4, 6}
    for file in tqdm.tqdm(files, total=len(files)):
        _, audio = wavfile.read(file.replace(".avi", ".wav"))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
        video = cv2.VideoCapture(file)
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            # Split into full-length windows and a possible shorter tail window.
            # Full windows all have the same shape so they can be stacked into one
            # GPU call; the tail (if any) is processed separately.
            n_full = sum(
                1 for i in range(batchSize)
                if (i + 1) * duration * 100 <= audioFeature.shape[0]
                and (i + 1) * duration * 25 <= videoFeature.shape[0]
            )
            scores = []
            if n_full > 0:
                bA = torch.FloatTensor(
                    numpy.stack([
                        audioFeature[i * duration * 100 : (i + 1) * duration * 100, :]
                        for i in range(n_full)
                    ])
                ).to(s.device)
                bV = torch.FloatTensor(
                    numpy.stack([
                        videoFeature[i * duration * 25 : (i + 1) * duration * 25, :, :]
                        for i in range(n_full)
                    ])
                ).to(s.device)
                with torch.no_grad():
                    eA = s.model.forward_audio_frontend(bA)
                    eV = s.model.forward_visual_frontend(bV)
                    eA, eV = s.model.forward_cross_attention(eA, eV)
                    out = s.model.forward_audio_visual_backend(eA, eV)
                    scores.extend(s.lossAV.forward(out, labels=None))
            # Handle the tail window (shorter than one full duration block)
            if n_full < batchSize:
                i = batchSize - 1
                inputA = torch.FloatTensor(
                    audioFeature[i * duration * 100 :, :]
                ).unsqueeze(0).to(s.device)
                inputV = torch.FloatTensor(
                    videoFeature[i * duration * 25 :, :, :]
                ).unsqueeze(0).to(s.device)
                with torch.no_grad():
                    eA = s.model.forward_audio_frontend(inputA)
                    eV = s.model.forward_visual_frontend(inputV)
                    eA, eV = s.model.forward_cross_attention(eA, eV)
                    out = s.model.forward_audio_visual_backend(eA, eV)
                    scores.extend(s.lossAV.forward(out, labels=None))
            allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(
            float
        )
        allScores.append(allScore)
    return allScores


def visualization(tracks, scores):
    # CPU: visulize the result for video format
    flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            # print(fidx, frame, len(flist))
            s = score[
                max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
            s = numpy.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = cv2.imread(flist[0])
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(pyaviPath, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)
    vOut.release()
    command = "ffmpeg -y -i %s -i %s -threads %d %s" % (
        os.path.join(pyaviPath, "video_only.avi"),
        os.path.join(pyaviPath, "audio.wav"),
        data_loader_thread,
        os.path.join(pyaviPath, "video_out.mp4"),
    )
    output = subprocess.call(command, shell=True, stdout=None)


def visualization_from_frames(tracks, scores, frames):
    faces = [[] for i in range(len(frames))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            # print(fidx, frame, len(flist))
            s = score[
                max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
            s = numpy.mean(s)
            faces[frame].append(
                {
                    "track": tidx,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                }
            )
    firstImage = frames[0]
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(pyaviPath, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
        image = frame
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        vOut.write(image)
    vOut.release()
    command = "ffmpeg -y -i %s -i %s -threads %d %s" % (
        os.path.join(pyaviPath, "video_only.avi"),
        os.path.join(pyaviPath, "audio.wav"),
        data_loader_thread,
        os.path.join(pyaviPath, "video_out.mp4"),
    )
    output = subprocess.call(command, shell=True, stdout=None)


def setup():
    # return initialized model
    s = talkNet()
    s.loadParameters(pretrained_model_path)
    sys.stderr.write(
        "Model %s loaded from previous state! \r\n" % (pretrained_model_path)
    )
    s.eval()

    DET = initialize_detector()

    return s, DET


# Main function
def main(
    s,
    DET,
    video_path,
    start_seconds=0,
    end_seconds=None,
    return_visualization=False,
    face_boxes="",
    in_memory_threshold=0,
    save_dir="/tmp/save",
):
    import subprocess
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    pyaviPath = os.path.join(save_dir, "pyavi")
    pyframesPath = os.path.join(save_dir, "pyframes")
    pyworkPath = os.path.join(save_dir, "pywork")
    pycropPath = os.path.join(save_dir, "pycrop")
    audioFilePath = os.path.join(pyaviPath, "audio.wav")

    # Initialization
    if os.path.exists(save_dir):
        rmtree(save_dir)
    os.makedirs(pyaviPath, exist_ok=True)
    os.makedirs(pyframesPath, exist_ok=True)
    os.makedirs(pyworkPath, exist_ok=True)
    os.makedirs(pycropPath, exist_ok=True)

    STORE_FRAMES_IN_MEMORY_THRESHOLD = in_memory_threshold

    # Extract video information
    # get video duration
    print("Extracting video information...")
    command = (
        "ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 '%s'"
        % video_path
    )
    video_num_frames = float(subprocess.check_output(command, shell=True))
    if video_num_frames == 0 or math.isnan(video_num_frames):
        raise ValueError("Video has no frames or is corrupted.")

    # get video length
    command = (
        "ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 '%s'"
        % video_path
    )
    video_duration = float(subprocess.check_output(command, shell=True))
    if video_duration == 0 or math.isnan(video_duration):
        raise ValueError("Video has no duration or is corrupted.")

    # get fps
    command = (
        "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 '%s'"
        % video_path
    )
    fps_output = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    fps = (
        float(fps_output.split("/")[0]) / float(fps_output.split("/")[1])
        if "/" in fps_output
        else float(fps_output)
    )
    if fps == 0 or math.isnan(fps):
        raise ValueError("Video has no frames or is corrupted.")

    print("Video duration: ", video_duration)
    print("FPS: ", fps)
    print("Video num frames: ", video_num_frames)

    if start_seconds < 0 or start_seconds > video_duration:
        raise ValueError("Start time is out of bounds.")
    if (
        end_seconds is not None
        and end_seconds != -1
        and (end_seconds < 0 or end_seconds > video_duration)
    ):
        end_seconds = video_duration
        print("End time is out of bounds. Setting to video duration.")

    if end_seconds is None or end_seconds == -1:
        end_seconds = video_duration

    # Extract video between start_seconds and end_seconds and save it to extracted_video.mp4
    start = start_seconds
    start_frame = int(start * fps)
    end_frame = int(end_seconds * fps)
    duration = end_seconds - start_seconds

    # Extract audio
    print("Extracting audio...")
    t = time.time()
    command = (
        "ffmpeg -y -i '%s' -ss %s -t %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic"
        % (video_path, start_seconds, duration, data_loader_thread, audioFilePath)
    )
    subprocess.call(command, shell=True, stdout=None)
    print("Audio extracted in %.3f seconds." % (time.time() - t))
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(audioFilePath))

    print("Extracting video frames...")
    t = time.time()
    # if less than 5000 frames, store in memory
    if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
        # use cv2 to extract frames
        import cv2

        vidcap = cv2.VideoCapture(video_path)
        frames = []
        frames_number_to_read = []
        for i in range(int((end_seconds - start_seconds) * 25) + 1):
            frame_number = int(start_frame + i * (fps / 25))
            if start_frame <= frame_number < end_frame:
                frames_number_to_read.append(frame_number)

        current_frame_number = start_frame
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        ret, current_frame = vidcap.read()
        if ret and current_frame is not None:
            print("Using OpenCV to extract frames...")
            count = 0
            for p in frames_number_to_read:
                if p == current_frame_number:
                    frames.append(current_frame)
                    count += 1
                else:
                    new_frame_number = current_frame_number
                    while new_frame_number < p:
                        ret, new_frame = vidcap.read()
                        if not ret:
                            break
                        new_frame_number += 1
                    if ret:
                        current_frame_number = p
                        current_frame = new_frame
                    frames.append(current_frame)
            vidcap.release()
        else:
            vidcap.release()
            try:
                print("Using imageio to extract frames...")
                import imageio

                frames = []
                cap = imageio.get_reader(video_path)
                if start_frame != 0:
                    cap.set_image_index(start_frame)

                print(f"load & seek time: {round(time.time() - t, 2)}s")
                current_frame = cap.get_next_data()
                current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)
                count = 0
                for p in frames_number_to_read:
                    frame_to_process = None
                    if p == current_frame_number:
                        frame_to_process = current_frame
                        frame_to_process = cv2.cvtColor(
                            frame_to_process, cv2.COLOR_RGB2BGR
                        )
                    else:
                        new_frame_number = p
                        if new_frame_number != current_frame_number + 1:
                            cap.set_image_index(new_frame_number)
                        try:
                            new_frame = cap.get_next_data()
                        except IndexError:
                            break
                        if new_frame is not None:
                            current_frame_number = p
                            current_frame = new_frame
                            frame_to_process = current_frame
                            frame_to_process = cv2.cvtColor(
                                frame_to_process, cv2.COLOR_RGB2BGR
                            )
                        else:
                            break
                    if frame_to_process is not None:
                        frames.append(frame_to_process)
                        count += 1
                print(f"Total frames processed: {count}")
                cap.close()
            except IndexError:
                try:
                    print("Failed...Using ffmpeg to extract frames...")
                    command = (
                        "ffmpeg -y -i '%s' -vf fps=25 -ss %s -t %s -threads %d -f image2 %s -loglevel panic"
                        % (
                            video_path,
                            start,
                            duration,
                            data_loader_thread,
                            os.path.join(pyframesPath, "%06d.jpg"),
                        )
                    )
                    subprocess.call(command, shell=True, stdout=None)
                    # read the frames from the extracted video frames
                    frames = [
                        cv2.imread(os.path.join(pyframesPath, "%06d.jpg" % i))
                        for i in range(int(duration * fps))
                    ]
                except Exception as e:
                    raise e
    print("Video frames extracted in %.3f seconds." % (time.time() - t))
    # Scene detection for the video frames
    print("Detecting scenes...")
    t = time.time()
    scene = scene_detect(
        video_path, save=False, start_frame=start_frame, end_frame=int(end_seconds * fps)
    )
    # convert scenes to 25 fps
    new_scenes = []
    for scene_data in scene:
        new_scenes.append(
            (
                FrameTimecode(
                    timecode=int(scene_data[0].frame_num * (25 / fps)), fps=25
                ),
                FrameTimecode(
                    timecode=int(scene_data[1].frame_num * (25 / fps)), fps=25
                ),
            )
        )
    print("Scenes detected in %.3f seconds." % (time.time() - t))
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(pyworkPath))
    # # Face detection for the video frames
    print("Detecting faces...")
    t = time.time()
    if face_boxes:
        face_boxes = face_boxes.strip().split("\n")
        faces = []
        for face_box in face_boxes:
            frame_data = face_box.split(",")
            frame_number = int(frame_data[0])
            frame_number -= start_frame
            # print(frame_number)
            frame_faces = []
            for i in range(1, len(frame_data), 5):
                bbox = [
                    float(frame_data[i]),
                    float(frame_data[i + 1]),
                    float(frame_data[i + 2]),
                    float(frame_data[i + 3]),
                ]
                conf = float(frame_data[i + 4])
                frame_faces.append({"frame": frame_number, "bbox": bbox, "conf": conf})
            faces.append(frame_faces)

        # these face boxes were from the original frames, so we need to convert them to the 25 fps frames
        # if the original fps is greater than 25, we need to convert the frame numbers to 25 fps and can sample the face boxes
        # if the original fps is less than 25, we need to interpolate the face boxes to 25 fps
        flist = glob.glob(os.path.join(pyframesPath, "*.jpg"))
        if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
            target_num_frames = len(frames)
        else:
            target_num_frames = len(flist)
        # print(f"Interpolating faces from {len(faces)} frames to {target_num_frames} frames...")
        new_faces = []
        for i in range(target_num_frames):
            target_frame = int(i * (fps / 25))
            # print(i, target_frame, start_frame, end_frame)
            if target_frame >= len(faces):
                new_faces.append(frame_faces)
                continue
            frame_faces = []
            for face in faces[target_frame]:
                frame_faces.append(
                    {
                        "frame": i,  # + (start_seconds * 25),
                        "bbox": face["bbox"],
                        "conf": face["conf"],
                    }
                )
            new_faces.append(frame_faces)
        faces = new_faces
    else:
        if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
            faces = predict_faces_from_frames(DET, frames)
        else:
            faces = predict_faces_streaming(DET, video_path, fps, end_seconds - start_seconds, start_seconds)

    print("Faces detected in %.3f seconds." % (time.time() - t))
    # sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(pyworkPath))

    # Face tracking
    print("Tracking faces...")
    t = time.time()
    allTracks, vidTracks = [], []
    first_frame = max(int(start_seconds * 25), 0)
    for shot in new_scenes:
        if (
            shot[1].frame_num - shot[0].frame_num >= min_track
        ):  # Discard the shot frames less than minTrack frames
            allTracks.extend(
                track_shot(
                    faces[
                        max(0, shot[0].frame_num - first_frame) : shot[1].frame_num
                        - first_frame
                    ]
                )
            )  # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    print("Faces tracked in %.3f seconds." % (time.time() - t))

    print("Cropping faces...")
    t = time.time()
    if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
        for ii, track in enumerate(allTracks):
            vidTracks.append(
                crop_video_from_frames(
                    track, os.path.join(pycropPath, "%05d" % ii), frames, audio_path=audioFilePath
                )
            )
    else:
        vidTracks = crop_video_from_video_all_tracks(allTracks, pycropPath, video_path, fps, audioFilePath)
    print("Faces cropped in %.3f seconds." % (time.time() - t))

    # Active Speaker Detection by TalkNet
    print("Detecting active speakers...")
    t = time.time()
    files = glob.glob("%s/*.avi" % pycropPath)
    files.sort()
    scores = evaluate_network(s, files)
    if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
        faces = [{"frame_number": i, "faces": []} for i in range(len(frames))]
    else:
        num_25fps = int((end_seconds - start_seconds) * 25) + 1
        faces = [{"frame_number": i, "faces": []} for i in range(num_25fps)]

    def get_scene_by_frame_number(frame_number):
        for scene_data in new_scenes:
            if (
                scene_data[0].frame_num <= frame_number
                and scene_data[1].frame_num > frame_number
            ):
                return scene_data
        return None

    # print('vidtracks', vidTracks)
    faces_by_scene = {}
    faces_by_frame = {}
    for tidx, track in enumerate(vidTracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            s = score[
                max(fidx - 3, 0) : min(fidx + 3, len(score) - 1)
            ]  # average smoothing
            s = numpy.mean(s)
            x1 = int(track["proc_track"]["x"][fidx] - track["proc_track"]["s"][fidx])
            y1 = int(track["proc_track"]["y"][fidx] - track["proc_track"]["s"][fidx])
            x2 = int(track["proc_track"]["x"][fidx] + track["proc_track"]["s"][fidx])
            y2 = int(track["proc_track"]["y"][fidx] + track["proc_track"]["s"][fidx])
            # print(frame, fidx, len(faces))
            faces[frame]["faces"].append(
                {
                    "track_id": tidx,
                    "raw_score": float(s),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "speaking": bool(s >= 0),
                }
            )
            if frame not in faces_by_frame:
                faces_by_frame[frame] = []
            faces_by_frame[frame].append(
                {
                    "track_id": tidx,
                    "raw_score": float(s),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "speaking": bool(s >= 0),
                }
            )
            # print(frame, frame + int(25 * start_seconds))
            scene_data = get_scene_by_frame_number(frame + int(25 * start_seconds))
            if scene_data is not None:
                scene_num = scene_data[0].frame_num
                if scene_num not in faces_by_scene:
                    faces_by_scene[scene_num] = {}
                # for each scene, keep a list of faces organized by frame number
                if frame not in faces_by_scene[scene_num]:
                    faces_by_scene[scene_num][frame] = []
                faces_by_scene[scene_num][frame].append(
                    {
                        "track_id": tidx,
                        "raw_score": float(s),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "speaking": bool(s >= 0),
                    }
                )

    target_num_frames = (int(end_seconds * fps) - int(start_seconds * fps)) + 1
    print(
        f"Interpolating faces from {len(faces)} frames to {target_num_frames} frames..."
    )

    interpolated_faces = []

    for scene_data in new_scenes:
        scene_num = scene_data[0].frame_num
        num_frames = scene_data[1].frame_num - scene_data[0].frame_num
        target_start_frame = int(scene_data[0].frame_num * (fps / 25))
        target_num_frames = int(num_frames * (fps / 25))
        # interpolate the faces in this scene
        interpolated_frames = []
        for i in range(target_start_frame, target_start_frame + target_num_frames + 1):
            frame_num = -1
            if target_num_frames > 0:
                frame_num = (
                    int(i * (num_frames / target_num_frames))
                    - new_scenes[0][0].frame_num
                )
            if frame_num not in faces_by_frame:
                frame_num = int(i * (25 / fps)) - new_scenes[0][0].frame_num
            interpolated_frames.append(
                {
                    "frame_number": i,
                    "faces": []
                    if frame_num not in faces_by_frame
                    else faces_by_frame[frame_num],
                }
            )
        interpolated_faces.extend(interpolated_frames)

    faces = interpolated_faces

    print("Active speakers detected in %.3f seconds." % (time.time() - t))
    if return_visualization:
        print("Visualizing the result...")
        t = time.time()

        if (end_seconds - start_seconds) * fps < STORE_FRAMES_IN_MEMORY_THRESHOLD:
            visualization_from_frames(vidTracks, scores, frames)
        else:
            visualization(vidTracks, scores)
        print("Result visualized in %.3f seconds." % (time.time() - t))
        # ffmpeg convert avi to mp4
        # subprocess.call(["ffmpeg", "-y", "-i", os.path.join(pyaviPath,'video_out.avi'), os.path.join(pyaviPath,'video_out.mp4')])
        return interpolated_faces, os.path.join(pyaviPath, "video_out.mp4")

    if "frames" in locals():
        del frames

    return interpolated_faces
