import math
import random
import cv2
import torch
import numpy as np
import onnx
import onnxruntime
from transformers import Wav2Vec2Processor, HubertModel, file_utils
from unet import Model
import logging

logger = logging.getLogger(__name__)

class AiModels:

    def __init__(self, checkpoint, asr="hubert", onnx_model=False, device="cuda:0") -> None:
        logger.info(f"Model's path : {file_utils.default_cache_path}")
        self._onnx_model = True if checkpoint is not None and checkpoint.endswith(".onnx") else False
        if(self._onnx_model):
            onnx_weights = onnx.load(checkpoint)
            onnx.checker.check_model(onnx_weights)
            providers = ["CUDAExecutionProvider"]
            self._net = onnxruntime.InferenceSession(checkpoint, providers=providers)
        else:
            self._net = Model(6, asr).cuda()
            self._net.load_state_dict(torch.load(checkpoint))
            self._net.eval()
        
        self._device=device
        logger.info("Loading the Wav2Vec2 Processor...")
        self._wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        
        logger.info("Loading the HuBERT Model...")
        self._hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        logger.info("Transferring the HuBERT Model to the device...")
        # logger.info(f"A. {self._hubert_model.dtype}")
        self._hubert_model.to(device)
        
        logger.info("Warming up wav2vec2 processor")
        self.warm_up()

    @torch.no_grad()
    def warm_up(self):
        # Create an array of zeros with the appropriate length
        num_samples = 16000 * 5
        speech = np.zeros(num_samples, dtype=np.int16)
        logger.info(f"2. {speech.shape}")
        input_values_all = self._wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000, padding=True).input_values # [1, T]
        logger.info(f"2. {input_values_all.shape}")
    @torch.no_grad()
    def create_audio_feature(self, speech, sample_rate=16000, length=1000, kernel=400, stride=320):
        # logger.info(f"0. {sample_rate}, {length}")
        logger.info(f"1. {speech.dtype}")
        if not np.issubdtype(speech.dtype, np.float64):
            speech = speech.astype(np.float64)
        if speech.ndim ==2:
            speech = speech[:, 0] # [T, 2] ==> [T,]

        logger.info(f"2. {speech.shape}")
        input_values_all = self._wav2vec2_processor(speech, return_tensors="pt", sampling_rate=sample_rate, padding=True).input_values # [1, T]
        logger.info(f"2. {input_values_all.shape}")
        logger.info(f"2. {input_values_all}")
        input_values_all = input_values_all.to(self._device)
        logger.info(f"3. {input_values_all.dtype}")
        # For long audio sequence, due to the memory limitation, we cannot process them in one run
        # HuBERT process the wav with a CNN of stride [5,2,2,2,2,2], making a stride of 320
        # Besides, the kernel is [10,3,3,3,3,2,2], making 400 a fundamental unit to get 1 time step.
        # So the CNN is euqal to a big Conv1D with kernel k=400 and stride s=320
        # We have the equation to calculate out time step: T = floor((t-k)/s)
        # To prevent overlap, we set each clip length of (K+S*(N-1)), where N is the expected length T of this clip
        # The start point of next clip should roll back with a length of (kernel-stride) so it is stride * N
       
        clip_length = stride * length
        logger.info(f"2.1 {kernel} -> {stride} -> {clip_length} -> {input_values_all.shape[0]} -> {input_values_all.shape[1]} -> {length}")
        
        num_iter = input_values_all.shape[1] // clip_length

        expected_T = math.ceil((input_values_all.shape[1] - (kernel-stride)) / stride)

        logger.info(f"2.2 {input_values_all.shape[1]} -> {length} -> {num_iter} iterations and expected_T is = {expected_T}")
        
        res_lst = []
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            # logger.info(f"start index : {start_idx} and end index : {end_idx}")
            input_values = input_values_all[:, start_idx: end_idx]
            # logger.info(f"4.a {input_values.dtype}")
            hidden_states = self._hubert_model.forward(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
            # logger.info(f"4.b {hidden_states.dtype}")
            res_lst.append(hidden_states[0])
        if num_iter > 0:
            input_values = input_values_all[:, clip_length * num_iter:]
        else:
            input_values = input_values_all
        # if input_values.shape[1] != 0:
        if input_values.shape[1] >= kernel: # if the last batch is shorter than kernel_size, skip it  
            # logger.info(f"4.c {input_values.dtype}")          
            hidden_states = self._hubert_model(input_values).last_hidden_state # [B=1, T=pts//320, hid=1024]
            # logger.info(f"4.d {hidden_states.dtype}")
            res_lst.append(hidden_states[0])
        ret = torch.cat(res_lst, dim=0).cpu() # [T, 1024]
        # assert ret.shape[0] == expected_T
        # assert abs(ret.shape[0] - expected_T) <= 1
        logger.info(f"5 Result length is {ret.shape[0]} and expected to be {expected_T}")
        if ret.shape[0] < expected_T:
            ret = torch.nn.functional.pad(ret, (0,0,0,expected_T-ret.shape[0]))
        else:
            ret = ret[:expected_T]
        
        logger.info(f"6. {ret.shape}")
        ret = self.make_even_first_dim(ret).reshape(-1, 2, 1024)
        logger.info(f"7. {ret.shape}")
        ret = ret.detach().numpy()
        logger.info(f"8. {ret.shape}")
        return ret
    
    @torch.no_grad()
    def create_video_frame(self, audio_features, video_frame):
        audio_feat = audio_features.reshape(32,32,32)
        audio_feat = audio_feat[None]
        
        # put those tensors in gpu memory
        if(self._onnx_model):
            audio_feat = audio_feat.cpu()
            video_frame = video_frame.cpu()
        else:
            audio_feat = audio_feat.cuda()
            video_frame = video_frame.cuda()
        
        """ logger.info(f"Video shape : {video_frame.dtype}")
        logger.info(f"Audio shape : {audio_feat.dtype}") """

        if(self._onnx_model):
            pred=self._net.run(None, {"input":video_frame.numpy(),"audio":audio_feat.numpy()})[0][0]
            pred = pred.transpose(1,2,0)*255
        else:
            with torch.no_grad():
                # with autocast():
                pred = self._net(video_frame, audio_feat)[0]
                pred = pred.cpu().numpy().transpose(1,2,0)*255
                
        pred = np.array(pred, dtype=np.uint8)
        
        return pred

    def make_even_first_dim(self, tensor):
        size = list(tensor.size())
        # logger.info(f"3.1. {size} size[0] % 2 == {size[0] % 2}")
        if size[0] % 2 == 1:
            size[0] -= 1
            tensor=tensor[:size[0]]
        # logger.info(f"3.2. {tensor.size()}")
        return tensor
    
    def get_audio_features(self, features, index=0):
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        
        return auds