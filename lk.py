import time
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)

from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import elevenlabs
import contextlib
import os
import cv2
import onnx
import onnxruntime
import torch
import numpy as np
from transformers import Wav2Vec2Processor, HubertModel
from unet import Model
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.livekit")

asr="hubert" 
dataset="./data/cansu" 
wav="./demo/etest.wav" 
checkpoint="./checkpoint/cansu/200.onnx"

mixed_precision = True

# Enable automatic mixed precision if available
try:
    if(mixed_precision):
        from torch.cuda.amp import autocast
    else:
        raise ImportError()
except ImportError:
    print("torch.cuda.amp not found. Mixed precision training will be disabled.")
    autocast = lambda: contextlib.nullcontext()






onnx_model = True if checkpoint is not None and checkpoint.endswith(".onnx") else False
audio_feat_path = wav.replace('.wav', '_hu.npy')

img_dir = os.path.join(dataset, "full_body_img/")
lms_dir = os.path.join(dataset, "landmarks/")
# len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

step_stride = 0
img_idx = 0
def prewarm(proc: JobProcess):
    start_time = time.time()
    print("Start loading models...")

    if(onnx_model):
        onnx_weights = onnx.load(checkpoint)
        onnx.checker.check_model(onnx_weights)
        providers = ["CUDAExecutionProvider"]
        net = onnxruntime.InferenceSession(checkpoint, providers=providers)
    else:
        net = Model(6, asr).cuda()
        net.load_state_dict(torch.load(checkpoint))
        net.eval()
    
    print("Loading the Wav2Vec2 Processor...")
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    print("Loading the HuBERT Model...")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

    end_time = time.time()  # Capture the end time
    execution_time = end_time - start_time  # Calculate the execution time
    
    print(f"Models loaded in {execution_time} seconds ")

async def entrypoint(ctx: JobContext):
    print("HELLO")
    return

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,

        ),
    )