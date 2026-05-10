import os
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from .box_utils import nms_
from .nets import S3FDNet

PATH_WEIGHT = os.path.join(os.path.dirname(__file__), "sfd_face.pth")
if os.path.isfile(PATH_WEIGHT) == False:
    cmd = (
        "wget -O %s https://storage.googleapis.com/mango-public-models/sfd_face.pth"
        % (PATH_WEIGHT)
    )
    subprocess.call(cmd, shell=True, stdout=None)
img_mean = np.array([104.0, 117.0, 123.0])[:, np.newaxis, np.newaxis].astype("float32")


class S3FD:
    def __init__(self, device="cuda"):

        tstamp = time.time()
        self.device = device

        # print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        PATH = os.path.join(os.getcwd(), PATH_WEIGHT)
        state_dict = torch.load(PATH, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        # print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))

    def _preprocess(self, image, scale):
        scaled = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled = np.swapaxes(scaled, 1, 2)
        scaled = np.swapaxes(scaled, 1, 0)
        scaled = scaled[[2, 1, 0], :, :].astype("float32")
        scaled -= img_mean
        scaled = scaled[[2, 1, 0], :, :]
        return scaled

    def detect_faces(self, image, conf_th=0.8, scales=[1]):
        results = self.detect_faces_batch([image], conf_th=conf_th, scales=scales)
        return results[0]

    def detect_faces_batch(self, images, conf_th=0.8, scales=[1]):
        """Run face detection on a batch of images in a single forward pass."""
        if not images:
            return []

        w, h = images[0].shape[1], images[0].shape[0]
        results = [np.empty(shape=(0, 5)) for _ in images]
        scale_tensor = torch.Tensor([w, h, w, h])

        with torch.no_grad():
            for s in scales:
                batch = torch.stack([
                    torch.from_numpy(self._preprocess(img, s)) for img in images
                ]).to(self.device)
                detections = self.net(batch).data  # [B, num_classes, num_dets, 5]

                for b in range(len(images)):
                    for i in range(detections.size(1)):
                        j = 0
                        while detections[b, i, j, 0] > conf_th:
                            score = detections[b, i, j, 0]
                            pt = (detections[b, i, j, 1:] * scale_tensor).cpu().numpy()
                            results[b] = np.vstack((results[b], (*pt, score)))
                            j += 1

        for b in range(len(images)):
            keep = nms_(results[b], 0.1)
            results[b] = results[b][keep]

        return results
