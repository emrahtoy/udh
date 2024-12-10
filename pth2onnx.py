import contextlib
from unet import Model
import onnx
import torch
import argparse
import onnxruntime
import numpy as np
import time

mixed_precision = False

# Enable automatic mixed precision if available
try:
    if(mixed_precision):
        from torch.cuda.amp import autocast, GradScaler
    else:
        raise ImportError()
except ImportError:
    print("torch.cuda.amp not found. Mixed precision training will be disabled.")
    autocast = lambda: contextlib.nullcontext()

parser = argparse.ArgumentParser(description='Train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--onnx_path', type=str, default="")     # end with .mp4 please
parser.add_argument('--checkpoint', type=str, default="")
parser.add_argument('--asr', type=str, default="hubert")

args = parser.parse_args()

def check_onnx(torch_out, img, audio):
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    providers = ["CUDAExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, providers=providers)
    print(ort_session.get_providers())
    ort_inputs = {ort_session.get_inputs()[0].name: img.cpu().numpy(), ort_session.get_inputs()[1].name: audio.cpu().numpy()}
    for i in range(1):
        t1 = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        t2 = time.time()
        print("onnx time cost::", t2 - t1)

    np.testing.assert_allclose(torch_out[0].cpu().numpy(), ort_outs[0][0], rtol=1e-03, atol=1e-05)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


net = Model(6).cuda()
net.load_state_dict(torch.load(args.checkpoint))
net.eval()
# hubert audio shape : torch.Size([1, 32, 32, 32])
# weber audio shape : torch.Size([1, 16, 32, 32])
# image shape : torch.Size([1, 6, 160, 160])
img = torch.zeros([1, 6, 160, 160]).cuda()
audio = torch.zeros([1, 32, 32, 32]).cuda() if args.asr=="hubert" else torch.zeros([1, 16, 32, 32]).cuda()

input_dict = {"input": img, "audio": audio}

with torch.no_grad():
    with autocast():
        torch_out = net(img, audio)
        # print(torch_out.shape)
        torch.onnx.export(net, (img, audio), args.onnx_path, input_names=['input', "audio"],
                        output_names=['output'], 
                        # dynamic_axes=dynamic_axes,
                        # example_outputs=torch_out,
                        opset_version=11,
                        export_params=True)
check_onnx(torch_out, img, audio)
