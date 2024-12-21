import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import AsyncIterable, Union
import cv2
from livekit import rtc
import numpy as np
import torch
from livekit.agents.tts import SynthesizedAudio
from livekit.agents import utils
from livekit_custom.agents.sync.ai_models import AiModels


logger = logging.getLogger(__name__)
@dataclass
class MediaInfo:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int
    audio_frame_size: int
    audio_frame_length: int

@dataclass
class Frame:
    video:rtc.VideoFrame
    audio:rtc.AudioFrame
class MediaStreamer:
    """Streams video and audio frames from a media file in an endless loop."""

    def __init__(self, data_path: Path, aiModels: AiModels, resize=1) -> None:
        self._stopped = False
        self._resize=resize
        self._aiModels = aiModels
        # fetch the first image to learn acutal width and height
        self._img_path = os.path.join(data_path, "full_body_img/")
        self._lms_path = os.path.join(data_path, "landmarks/")
        self._img_len = 300 #len(os.listdir(self._img_path)) - 1
        exm_img = cv2.imread(self._img_path+"0.jpg")

        h, w = exm_img.shape[:2]
        if(self._resize>1):
            h= int(h / self._resize)
            w= int(w / self._resize)
        
        
        self._info = MediaInfo(
            video_width=w,
            video_height=h,
            video_fps=float(25),  # type: ignore
            audio_sample_rate=16000,
            audio_channels=1,
            audio_frame_size=int(16000/25), # 40ms
            audio_frame_length=int(1000/25)
        )

        logger.info(f"Media info : {self._info}")

        self._silence = None
        self._index = 0
        self._step_stride = 1
        # self._audio_chunks = asyncio.Queue() # to be used for calculating audio frames as well as video frames
        self._audio_features = None # to be used for holding audio features
        self._rendered_audio_buffer = asyncio.Queue()
        self._rendered_video_buffer = asyncio.Queue()
        self._audio_buffer = bytearray() # to be used to fetch audio features
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._switchAt = None
        self._frame_cache=50
        self._switch=False
        audio_frame = rtc.AudioFrame.create(self._info.audio_sample_rate, self._info.audio_channels, self._info.audio_frame_size)
        self._silence = np.frombuffer(audio_frame.data, dtype=np.int16)
        self._temp_buffer = np.array([], dtype=np.int16)
        self.warmup()
    
    def warmup(self):
        self._images=[]
        self._tensors=[]
        self._landmarks=[]

        self.cache_images()
        self.cache_landmarks()
        self.cache_tensors()
    @property
    def info(self) -> MediaInfo:
        return self._info
    async def stream(self) -> AsyncIterable[Frame]:
        """Streams image frames from the image list in an endless loop."""
        while not self._stopped:
            if self._stopped:
                break
            #this will do the rewind when needed
            if self._index==self._img_len-1:
                self._step_stride = -1
            if self._index == 0:
                self._step_stride = 1
            
            if(self._switchAt is not None and self._index == self._switchAt and self._switch==False):
                self._switch=True
                logger.info(f"Switched at {self._index} because it is set {self._switchAt}")
            
            """ qsize = self._rendered_video_buffer.qsize()
            if(qsize>0):
                logger.info(f"Buffered frames : {self._rendered_video_buffer.qsize()}") """

            if self._switch and self._rendered_video_buffer.empty()==False:
                self._switchAt = None

                try: 
                    video = self._rendered_video_buffer.get_nowait()
                except Exception as e:
                    video = None

                try:
                    audio = self._rendered_audio_buffer.get_nowait()
                except Exception as e:
                    audio = None
                
                yield Frame(video=video,audio=audio)
                
            else:
                self._switch = False
                video_frame = self._images[self._index]
                audio_frame = self._silence

                yield Frame(video=rtc.VideoFrame(
                    width=video_frame.shape[1],
                    height=video_frame.shape[0],
                    type=rtc.VideoBufferType.RGB24,
                    data=cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB).tobytes(),
                ),audio=rtc.AudioFrame(
                    data=audio_frame.tobytes(),
                    sample_rate=self.info.audio_sample_rate,
                    num_channels=self._info.audio_channels,
                    samples_per_channel=self._info.audio_frame_size,
                ))
                

            self._index=self._index + self._step_stride
            await asyncio.sleep(0)
    async def stream_video(self) -> AsyncIterable[rtc.VideoFrame]:
        """Streams image frames from the image list in an endless loop."""
        while not self._stopped:
            if self._stopped:
                break
            #this will do the rewind when needed
            if self._index==self._img_len-1:
                self._step_stride = -1
            if self._index == 0:
                self._step_stride = 1
                
            if(self._switchAt is not None and self._index == self._switchAt and self._switch==False):
                self._switch=True
                logger.info(f"Switched at {self._index} because it is set {self._switchAt}")
            
            if self._switch and self._rendered_video_buffer.empty()==False:
                self._switchAt = None
                yield self._rendered_video_buffer.get_nowait()
            else:
                self._switch = False
                frame = self._images[self._index]
                yield rtc.VideoFrame(
                    width=frame.shape[1],
                    height=frame.shape[0],
                    type=rtc.VideoBufferType.RGB24,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes(),
                )
                

            self._index=self._index + self._step_stride
    async def stream_audio(self) -> AsyncIterable[rtc.AudioFrame]:
        """Streams audio frames from the media file in an endless loop."""
        if self._silence is None:
            audio_frame = rtc.AudioFrame.create(self._info.audio_sample_rate, self._info.audio_channels, self._info.audio_frame_size)
            self._silence = np.frombuffer(audio_frame.data, dtype=np.int16)
        while not self._stopped:
            frame = None
            if self._stopped:
                break

            if self._switch and self._rendered_audio_buffer.empty()==False:
                yield self._rendered_audio_buffer.get_nowait()
            else:
                # creating silent frames for idling
                frame = self._silence
                yield rtc.AudioFrame(
                    data=frame.tobytes(),
                    sample_rate=self.info.audio_sample_rate,
                    num_channels=self._info.audio_channels,
                    samples_per_channel=self._info.audio_frame_size,
                )
                

    
    async def speak(self,text:str,tts)->None:
        # logger.info(f"Going to tts : {text}")
        self._audio_buffer = bytearray()
        async for output in tts.synthesize(text):
            await self.put_audio_chunks(output)
        self.put_audio_chunks(None)

        audio_bytes = np.frombuffer(self._audio_buffer, dtype=np.int16)
        # self._audio_features = await self._aiModels.create_audio_feature(audio_bytes,self._info.audio_sample_rate, 1000)

        try:
            audio_features = asyncio.get_event_loop().run_in_executor(
                self._executor, 
                self._aiModels.create_audio_feature, 
                audio_bytes,
                self._info.audio_sample_rate,
                1000
            )
            self._audio_features = await audio_features
        except TimeoutError:
            logger.error('The coroutine took too long, cancelling the task...')
            audio_features.cancel()
        except Exception as e:
            # Handle the exception here
            print(f"An error occurred: {e}")
        
        asyncio.create_task(self.render_frame())

    async def render_frame(self)->None:
        # logger.info(f"{str(self._index)} Image frame processing now")
        jump_index = (self._index + self._frame_cache) % self._img_len
        step_stride = self._step_stride
        self._switchAt = jump_index + 0 # to be sure it doesnt change in the loop below
        logger.info(f"index is now {self._index} and will be switched at {self._switchAt}")
        
        audio_sent=False
        type=1
        for i in range(self._audio_features.shape[0]):
            # start_time = time.perf_counter_ns()
            # this will do the rewind when needed
            if jump_index==self._img_len-1:
                step_stride = -1
            if jump_index == 0:
                step_stride = 1
            
            # logger.debug(f"{len(self._audio_buffer)} -> {len(audio_bytes)}")
            audio_features = self._aiModels.get_audio_features(self._audio_features,i)

            if(type==1):
                start_idx=i*1280
                end_idx=start_idx+1280
                audio_bytes = self._audio_buffer[start_idx:end_idx]
                video_frame = asyncio.create_task(self.create_video_frame(audio_features, self._tensors[jump_index], self._images[jump_index]))
                audio_frame = asyncio.create_task(self.create_audio_frame(audio_bytes))
                await asyncio.gather(video_frame,audio_frame)
            else:
                if(audio_sent==False):
                    video_frame = asyncio.create_task(self.create_video_frame(audio_features, self._tensors[jump_index], self._images[jump_index]))
                    audio_frame = asyncio.create_task(self.create_audio_frame(self._audio_buffer))
                    audio_sent=True
                    await asyncio.gather(video_frame,audio_frame)
                else:
                    video_frame = asyncio.create_task(self.create_video_frame(audio_features, self._tensors[jump_index], self._images[jump_index]))
                    await video_frame

            jump_index=jump_index + step_stride
            """ end_time = time.perf_counter_ns()
            logger.warning(f'{i} capture frame : {(end_time-start_time)/1e6:.3f}') """
            
    async def put_audio_chunks(self,synthesizedAudio:SynthesizedAudio,to:str="buffer") -> None:
        
        if(synthesizedAudio is None):
            current_chunk = self._temp_buffer.copy()
            self._temp_buffer = self._temp_buffer = np.array([], dtype=np.int16)
        else:
            current_chunk = np.frombuffer(synthesizedAudio.frame.data, dtype=np.int16) # Source samples per channel is 1600 

        if(self._temp_buffer.size>0):
            current_chunk = np.concatenate((self._temp_buffer, current_chunk))
            self._temp_buffer = np.array([], dtype=np.int16)

        size = np.size(current_chunk)
        target_samples_per_channel = self._info.audio_frame_size # current value is 640
        
        while size >= target_samples_per_channel:
            # Extract a frame from the current chunk
            

            audio_frame = current_chunk[:target_samples_per_channel]
            current_chunk = current_chunk[target_samples_per_channel:]

            # push frame to queue
            if(to == "buffer"):
                self._audio_buffer.extend(audio_frame.tobytes())
            else:
                await self._audio_chunks.put(audio_frame)
            size=size-target_samples_per_channel
        
        if np.size(current_chunk) < target_samples_per_channel and np.size(current_chunk) > 0:
            if(synthesizedAudio is None):
                audio_frame= np.pad(current_chunk, (0, target_samples_per_channel - np.size(current_chunk)), mode='constant')
                if(to == "buffer"):
                    self._audio_buffer.extend(audio_frame.tobytes())
                else:
                    await self._audio_chunks.put(audio_frame)
            else:
                self._temp_buffer = current_chunk.copy()
    def aclose(self) -> None:
        """Closes the media container and stops streaming."""
        self._stopped = True

    async def create_audio_frame(self,audio_bytes):
        # logger.debug(f"Creating audio {len(audio_bytes)}")
        self._rendered_audio_buffer.put_nowait(rtc.AudioFrame(
                data=audio_bytes,
                sample_rate=self.info.audio_sample_rate,
                num_channels=self._info.audio_channels,
                samples_per_channel=self._info.audio_frame_size,
            ))
    async def create_video_frame(self, audio_features, cached_tensor, cached_image):
        try:
            processed_frame_piece_future = asyncio.get_event_loop().run_in_executor(
                self._executor, 
                self._aiModels.create_video_frame, 
                audio_features,
                cached_tensor[0]
            )
            processed_frame_piece = await processed_frame_piece_future
        except TimeoutError:
            logger.error('The coroutine took too long, cancelling the task...')
            audio_features.cancel()
        except Exception as e:
            # Handle the exception here
            print(f"An error occurred: {e}")


        #stitch the processed frame piece

        #merge prediction and original image
        crop_img_ori = cached_tensor[1]
        crop_img_ori[4:164, 4:164] = processed_frame_piece

        #resize again ( why )
        crop_img_ori = cv2.resize(crop_img_ori, (cached_tensor[6], cached_tensor[7]))
        
        #merge into the original image (frame)
        img = cached_image.copy()
        img[cached_tensor[4]:cached_tensor[5], cached_tensor[2]:cached_tensor[3]] = crop_img_ori
        
        self._rendered_video_buffer.put_nowait(rtc.VideoFrame(
                    width=img.shape[1],
                    height=img.shape[0],
                    type=rtc.VideoBufferType.RGB24,
                    data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tobytes(),
                ))
        
    def cache_images(self):
        # reading images
        for i in range(self._img_len):
            img = cv2.imread(self._img_path+str(i)+".jpg")
            if(self._resize>1):
                img = cv2.resize(img, (self._info.video_width, self._info.video_height), cv2.INTER_AREA)
            # self._images.append(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))
            self._images.append(img)

    def cache_landmarks(self):
        # reading landmarks
        
        for i in range(self._img_len):
            with open(self._lms_path+ str(i)+'.lms', "r") as f:
                lms_list = []
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            self._landmarks.append(np.array(lms_list, dtype=np.int32))

    def cache_tensors(self):
        for img_idx in range(self._img_len):
            # reading landmark and image
            lms = self._landmarks[img_idx]
            img = self._images[img_idx]

            #determining width, height and positions
            xmin = lms[1][0]
            ymin = lms[52][1]

            xmax = lms[31][0]

            if(self._resize>1):
                ymin = int(ymin / self._resize)
                xmin = int(xmin / self._resize)
                xmax = int(xmax / self._resize)

            width = xmax - xmin
            ymax = ymin + width
            height = ymax - ymin

            #get a crop from original image and determine its size
            crop_img = img[ymin:ymax, xmin:xmax]
            # h, w = crop_img.shape[:2] # isnt this same with the width and height above ?

            #get resized version of crop_img
            crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)

            img_real_ex = crop_img[4:164, 4:164].copy()
            img_real_ex_ori = img_real_ex.copy()

            #prepare masked image from the img_real_ex_ori
            img_masked = cv2.rectangle(img_real_ex_ori,(5,5,150,145),(0,0,0),-1)
            
            # turn/flip the images
            img_masked = img_masked.transpose(2,0,1).astype(np.float32)
            img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)

            # merge masked and get tensor version of it
            img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
            img_masked_T = torch.from_numpy(img_masked / 255.0)
            img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]
            
            # logger.info(f"{img_idx}. The numpy shape is {img_real_ex.shape} + {img_masked.shape} = {img_concat_T.numpy().shape}")
            # logger.info(f"{img_idx}. The tensor to numpy shape is {img_real_ex_T.numpy().shape} + {img_masked_T.numpy().shape} = {img_concat_T.numpy().shape}")
            
            cache=[img_concat_T,crop_img,xmin,xmax,ymin,ymax,width,height]
            self._tensors.append(cache)