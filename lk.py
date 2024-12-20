from asyncio import Event
import os
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
from livekit.plugins import elevenlabs, azure
import contextlib

import torch
import numpy as np

from livekit_custom.agents.sync.ai_models import AiModels
from dotenv import load_dotenv
from livekit.agents.tts import SynthesizedAudio, ChunkedStream
from typing import AsyncIterable
from livekit_custom.agents.sync.media_streamer import Frame, MediaStreamer
from livekit_custom.agents.sync.av_sync import AVSynchronizer


load_dotenv(dotenv_path=".env.livekit")

logger = logging.getLogger(__name__)

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

aiModels = None
tts = None
streamer = None
media_info = None
def prewarm(proc: JobProcess):
    logger.info("WARMING UP!")
    # asyncio.run(warm_up())
    logger.info("WARM AND READY!")
    
def create_streamer(media_path, aiModels)->MediaStreamer :
    return MediaStreamer(media_path, aiModels, 1)

def create_aimodels(checkpoint)->AiModels:
    return AiModels(checkpoint=checkpoint)

async def warm_up()->None:
    global tts, streamer, media_info, aiModels

    start_time = time.time()
    if(aiModels is None):
        logger.info("Warming up AI Models...")
        aiModels = await asyncio.get_event_loop().run_in_executor(None, create_aimodels,"./checkpoint/mesut/200.pth")
    else:
        logger.info("AI Models already initialized")

    end_time = time.time()  # Capture the end time
    execution_time = end_time - start_time  # Calculate the execution time
        
    logger.info(f"AI Models warmed up in {execution_time} seconds ")

    start_time = time.time()
    logger.info("Warming up Streamer...")
    # Create media streamer
    # Should we add a sample video file?
    media_path = Path("./data/mesut/")
    if(streamer is None):
        streamer = await asyncio.get_event_loop().run_in_executor(None, create_streamer, media_path, aiModels)
        streamer.warmup()
        media_info = streamer.info
    else:
        logger.info("Streamer already initialized")

    # Create TTS service
    if( tts is None):
        tts = elevenlabs.TTS(chunk_length_schedule=500, language="tr", encoding="pcm_16000",api_key=os.environ.get("11LABS_API_KEY"))
        # tts = azure.TTS(sample_rate=16000, language="tr_TR")
    else:
        logger.info("TTS service already initialized")
    
    end_time = time.time()  # Capture the end time
    execution_time = end_time - start_time  # Calculate the execution time
    logger.info(f"Streamer warmed up in {execution_time} seconds ")
async def my_shutdown_hook():
    logger.info("Shutting down the agent")
    global aiModels
    global streamer
    aiModels = None
    streamer = None
    logger.info("Agent shut down")

@utils.log_exceptions(logger=logger)
async def _direct_push_stream_frames(
    stream: AsyncIterable[Frame], audio_source: rtc.AudioSource, video_source:rtc.VideoSource
) -> None:
    global streamer
    async for frame in stream:
        # logger.info(f"{audio_source} - {video_source} - {frame.audio} - {frame.video}")
        await audio_source.capture_frame(frame.audio)
        video_source.capture_frame(frame.video)
@utils.log_exceptions(logger=logger)
async def _speak(text:str, wait=0):
    global streamer
    await asyncio.sleep(wait)
    logger.info("Speaking : "+text)
    asyncio.create_task(streamer.speak(text=text,tts=tts))

async def entrypoint(job: JobContext):
    global tts, streamer, media_info
    await job.connect()
    room = job.room
    
    # wait for the first participant to arrive
    logger.info("Awaiting for the participant")
    await job.wait_for_participant()
    await warm_up()
    
    # Create video and audio sources/tracks
    video_source = rtc.VideoSource(
        width=media_info.video_width,
        height=media_info.video_height
    )
    audio_source = rtc.AudioSource(
        sample_rate=media_info.audio_sample_rate,
        num_channels=media_info.audio_channels,
    )

    # Publish tracks
    video_track = rtc.LocalVideoTrack.create_video_track("video", video_source)
    video_options = rtc.TrackPublishOptions(
        # since the agent is a participant, our video I/O is its "camera"
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=False,
        # when modifying encoding options, max_framerate and max_bitrate must both be set
        video_encoding=rtc.VideoEncoding(
            max_framerate=25,
            max_bitrate=3_000_000,
        ),
        video_codec=rtc.VideoCodec.H264
    )
    audio_track = rtc.LocalAudioTrack.create_audio_track("audio", audio_source)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)

    video_publication = await room.local_participant.publish_track(video_track, video_options)
    audio_publication = await room.local_participant.publish_track(audio_track, audio_options)

    await video_publication.wait_for_subscription()
    await audio_publication.wait_for_subscription()

    @room.on("participant_disconnected")
    def terminate():
        global streamer
        streamer.aclose()
    try:
        
        stream = streamer.stream()
        stream_task = asyncio.create_task(_direct_push_stream_frames(stream, audio_source, video_source))
        speak_task = asyncio.create_task(_speak("Oldum olası bu dünyada bir karara varamadım.",2))
        await stream_task

    finally:
        await my_shutdown_hook()
        # disconnect from the room
        job.shutdown(reason="Session ended")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            job_memory_limit_mb=0,
            job_memory_warn_mb=8000,
            initialize_process_timeout=25,
            num_idle_processes=0
        ),
    )