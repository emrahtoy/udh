from dataclasses import dataclass
import os
from pathlib import Path
from typing import AsyncIterable
import cv2
from livekit import rtc
import numpy as np
import torch

@dataclass
class MediaInfo:
    video_width: int
    video_height: int
    video_fps: float
    audio_sample_rate: int
    audio_channels: int
    
class MediaStreamer:
    """Streams video and audio frames from a media file in an endless loop."""

    def __init__(self, data_path: Path) -> None:
        self._stopped = False

        # fetch the first image to learn acutal width and height
        self._img_path = os.path.join(data_path, "full_body_img/")
        self._lms_path = os.path.join(data_path, "landmarks/")
        self._img_len = 250 #len(os.listdir(self._img_path)) - 1
        exm_img = cv2.imread(self._img_path+"0.jpg")

        h, w = exm_img.shape[:2]

        self._images=[]
        self._tensors=[]
        self._landmarks=[]
        
        self._info = MediaInfo(
            video_width=w,
            video_height=h,
            video_fps=float(25),  # type: ignore
            audio_sample_rate=16000,
            audio_channels=1,
        )

        self.cache_images()
        self.cache_landmarks()
        self.cache_tensors()
        self._index = 0
    
    @property
    def info(self) -> MediaInfo:
        return self._info
    async def stream_video(self) -> AsyncIterable[rtc.VideoFrame]:
        """Streams image frames from the image list in an endless loop."""
        while not self._stopped:
            if self._stopped:
                break
            #this will do the rewind when needed
            if self._index==self._img_len-1:
                step_stride = -1
            if self._index == 0:
                step_stride = 1
                
            frame = self._images[self._index]
            yield rtc.VideoFrame(
                width=frame.shape[1],
                height=frame.shape[0],
                type=rtc.VideoBufferType.BGRA,
                data=frame.tobytes(),
            )

            self._index=self._index + step_stride

    async def stream_audio(self) -> AsyncIterable[rtc.AudioFrame]:
        """Streams audio frames from the media file in an endless loop."""
        while not self._stopped:
            if self._stopped:
                break
            # creating silent frames for testing purposes
            audio_frame = rtc.AudioFrame.create(self._info.audio_sample_rate, self._info.audio_channels, self._info.audio_sample_rate // 25)
            frame = np.frombuffer(audio_frame.data, dtype=np.int16)

            # Convert audio frame to raw int16 samples
            #frame = (1 * 32768).astype(np.int16)
            yield rtc.AudioFrame(
                data=frame.tobytes(),
                sample_rate=self.info.audio_sample_rate,
                num_channels=self._info.audio_channels,
                samples_per_channel=self._info.audio_sample_rate // 25,
            )

    async def aclose(self) -> None:
        """Closes the media container and stops streaming."""
        self._stopped = True

    def cache_images(self):
        # reading images
        for i in range(self._img_len):
            img = cv2.imread(self._img_path+str(i)+".jpg")
            self._images.append(cv2.cvtColor(img, cv2.COLOR_BGR2BGRA))

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
            cache=[img_concat_T,crop_img,xmin,xmax,ymin,ymax,width,height]
            self._tensors.append(cache)