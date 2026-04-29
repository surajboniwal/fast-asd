import sys
from fast_asd.talknet import demoTalkNet
from fast_asd.talknet.talkNet import talkNet

s = talkNet()
s.loadParameters(demoTalkNet.pretrained_model_path)
s.eval()

DET = demoTalkNet.initialize_detector(device="cpu")

results = demoTalkNet.main(
    s,
    DET,
    video_path="output2.mp4",
    start_seconds=10,
    end_seconds=20,
    return_visualization=False,
)

for frame in results[:5]:
    print(frame)

print("Output video: save/pyavi/video_out.mp4")
