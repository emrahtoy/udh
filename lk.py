import time
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit import api, rtc
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

mixed_precision = False

# Enable automatic mixed precision if available
try:
    if(mixed_precision):
        from torch.cuda.amp import autocast
    else:
        raise ImportError()
except ImportError:
    print("torch.cuda.amp not found. Mixed precision training will be disabled.")
    autocast = lambda: contextlib.nullcontext()


torch.device("cuda",0)



onnx_model = True if checkpoint is not None and checkpoint.endswith(".onnx") else False
audio_feat_path = wav.replace('.wav', '_hu.npy')

img_dir = os.path.join(dataset, "full_body_img/")
lms_dir = os.path.join(dataset, "landmarks/")
# len_img = len(os.listdir(img_dir)) - 1
exm_img = cv2.imread(img_dir+"0.jpg")
h, w = exm_img.shape[:2]

step_stride = 0
img_idx = 0
net = None
onnx_weights = None

SAMPLE_RATE = 16000
NUM_CHANNELS = 1 # mono audio
AMPLITUDE = 2 ** 8 - 1
SAMPLES_PER_CHANNEL = 160 # 10ms at 48kHz

WIDTH = h
HEIGHT = w

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
    hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to("cuda:0")

    end_time = time.time()  # Capture the end time
    execution_time = end_time - start_time  # Calculate the execution time
    
    print(f"Models loaded in {execution_time} seconds ")

async def entrypoint(ctx: JobContext):
    print("HELLO")
    await ctx.connect(
        # valid values are SUBSCRIBE_ALL, SUBSCRIBE_NONE, VIDEO_ONLY, AUDIO_ONLY
        # when omitted, it defaults to SUBSCRIBE_ALL
        auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE,
    )

    # disconnect from the room
    async def my_shutdown_hook():
        if(onnx_model):
            onnx_weights = None
        net = None
        exm_img = None
        print("Memory cleaned")

    ctx.add_shutdown_callback(my_shutdown_hook)

    video_source = rtc.VideoSource(WIDTH, HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("Camera", video_source)
    options = rtc.TrackPublishOptions(
        # since the agent is a participant, our video I/O is its "camera"
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_encoding=rtc.VideoEncoding(max_framerate=25),
        video_codec=rtc.VideoCodec.H264,
    )
    audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
    audio_track = rtc.LocalAudioTrack.create_audio_track("example-track", audio_source)
    # since the agent is a participant, our audio I/O is its "microphone"
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE,
        audio_encoding=rtc.AudioEncoding(max_bitrate=16000)
        )
    # ctx.agent is an alias for ctx.room.local_participant
    publication = await ctx.agent.publish_track(audio_track, options)
    publication = await ctx.agent.publish_track(video_track, options)
    
    
    """ api_client = api.LiveKitAPI(
        os.getenv("LIVEKIT_URL"),
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET"),
    )
    await api_client.room.delete_room(api.DeleteRoomRequest(ctx.job.room.name)) """
    ctx.shutdown(reason="Session ended")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,

        ),
    )