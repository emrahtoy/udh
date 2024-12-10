from asyncio import Event
from pathlib import Path
import asyncio

import logging
import time
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    utils
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
from livekit.agents.tts import SynthesizedAudio, ChunkedStream
from typing import AsyncIterable
from livekit_custom.agents.sync.media_streamer import MediaStreamer
from livekit_custom.agents.sync.av_sync import AVSynchronizer


load_dotenv(dotenv_path=".env.livekit")

logger = logging.getLogger(__name__)

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
FRAMES_PER_SECOND = 25
FRAMES_SIZE = SAMPLE_RATE // FRAMES_PER_SECOND  # 640 samples per frame

WIDTH = h
HEIGHT = w
wav2vec2_processor = None
hubert_model = None
tts = None
def prewarm(proc: JobProcess):
    return
    global hubert_model
    global wav2vec2_processor
    global net
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
    tts = elevenlabs.TTS(chunk_length_schedule=64, language="tr", encoding="pcm_16000",api_key="sk_6b8d75ee6770f09c73b21e4b1d3b80e4f7a31e0ea4f13acb")
async def my_shutdown_hook():
    global hubert_model
    global wav2vec2_processor
    global onnx_model
    global net
    global exm_img

    if(onnx_model):
        onnx_weights = None
    net = None
    exm_img = None
    hubert_model = None
    wav2vec2_processor = None
    print("Memory cleaned")

async def entrypoint(ctx: JobContext):
    print("HELLO")
    global tts
    await ctx.connect(
        # valid values are SUBSCRIBE_ALL, SUBSCRIBE_NONE, VIDEO_ONLY, AUDIO_ONLY
        # when omitted, it defaults to SUBSCRIBE_ALL
        auto_subscribe=AutoSubscribe.SUBSCRIBE_NONE,
    )

    # disconnect from the room
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
    audio_track = rtc.LocalAudioTrack.create_audio_track("Microphone", audio_source)
    # since the agent is a participant, our audio I/O is its "microphone"
    options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_MICROPHONE
        )
    # ctx.agent is an alias for ctx.room.local_participant
    publication_audio = await ctx.agent.publish_track(audio_track, options)
    publication_video = await ctx.agent.publish_track(video_track, options)

    
    async for output in tts.synthesize("Merhaba Dünya"):
        await process_audio_stream(output)
    
    """ api_client = api.LiveKitAPI(
        os.getenv("LIVEKIT_URL"),
        os.getenv("LIVEKIT_API_KEY"),
        os.getenv("LIVEKIT_API_SECRET"),
    )
    await api_client.room.delete_room(api.DeleteRoomRequest(ctx.job.room.name)) """
    ctx.shutdown(reason="Session ended")

async def entrypoint2(job: JobContext):
    await job.connect()
    room = job.room

    # Create media streamer
    # Should we add a sample video file?
    media_path = Path("./data/mesut/")
    streamer = MediaStreamer(media_path)
    media_info = streamer.info

    # Create video and audio sources/tracks
    queue_size_ms = 1000  # 1 second
    video_source = rtc.VideoSource(
        width=media_info.video_width,
        height=media_info.video_height,
    )
    print(media_info)
    audio_source = rtc.AudioSource(
        sample_rate=media_info.audio_sample_rate,
        num_channels=media_info.audio_channels,
        queue_size_ms=queue_size_ms,
    )

    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)

    # Publish tracks
    video_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    await room.local_participant.publish_track(video_track, video_options)
    await room.local_participant.publish_track(audio_track, audio_options)

    @utils.log_exceptions(logger=logger)
    async def _push_video_frames(
        video_stream: AsyncIterable[rtc.VideoFrame], av_sync: AVSynchronizer
    ) -> None:
        """Task to push video frames to the AV synchronizer."""
        async for frame in video_stream:
            await av_sync.push(frame)
            await asyncio.sleep(0)

    @utils.log_exceptions(logger=logger)
    async def _push_audio_frames(
        audio_stream: AsyncIterable[rtc.AudioFrame], av_sync: AVSynchronizer
    ) -> None:
        """Task to push audio frames to the AV synchronizer."""
        async for frame in audio_stream:
            await av_sync.push(frame)
            await asyncio.sleep(0)

    try:
        av_sync = AVSynchronizer(
            audio_source=audio_source,
            video_source=video_source,
            video_fps=media_info.video_fps,
            video_queue_size_ms=queue_size_ms,
        )

        # Create and run video and audio streaming tasks
        video_stream = streamer.stream_video()
        audio_stream = streamer.stream_audio()

        video_task = asyncio.create_task(_push_video_frames(video_stream, av_sync))
        audio_task = asyncio.create_task(_push_audio_frames(audio_stream, av_sync))

        # Wait for both tasks to complete
        await asyncio.gather(video_task, audio_task)
        await av_sync.wait_for_playout()

    finally:
        await streamer.aclose()
        await av_sync.aclose()
async def process_audio_stream(synthesizedAudio:SynthesizedAudio, callback=None, cancelEvent:Event=None):
    global hubert_model
    res_lst = []
    num_iter = 0
    print(f"Frame length : {len(synthesizedAudio.frame.to_wav_bytes())}")
    current_chunk = np.frombuffer(synthesizedAudio.frame.to_wav_bytes(), dtype=np.int16)
    
    while np.size(current_chunk) >= FRAMES_SIZE:
        # Extract a frame from the current chunk
        frame = current_chunk[:FRAMES_SIZE]
        current_chunk = current_chunk[FRAMES_SIZE:]

        # Convert frame to input values
        input_values = wav2vec2_processor(frame, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
        input_values = input_values.float().to("cuda:0")

        # Process the frame
        hidden_states = hubert_model(input_values).last_hidden_state
        res_lst.append(hidden_states[0])
        num_iter += 1

    # Process any remaining samples if they form a complete frame
    if np.size(current_chunk) < FRAMES_SIZE:
        input_values = wav2vec2_processor(np.pad(current_chunk, FRAMES_SIZE), return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
        input_values = input_values.float().to("cuda:0")
        hidden_states = hubert_model(input_values).last_hidden_state
        res_lst.append(hidden_states[0])
        num_iter += 1
    # Concatenate the results

    ret = torch.cat(res_lst, dim=0).cpu()
    if callback is not None:
        callback(ret)
    else:
        print(f"{num_iter} iteration processed")
    return ret

async def fetch_frame(state="Idle",frame=0, audio=None):
    print(frame)

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint2,
            prewarm_fnc=prewarm,

        ),
    )