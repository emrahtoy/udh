from transformers import Wav2Vec2Processor, HubertModel
import numpy as np
import torch

# Initialize the processor and model
print("Loading the Wav2Vec2 Processor...")
wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
print("Loading the HuBERT Model...")
hubert_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")

# Set the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
hubert_model = hubert_model.to(device)

# Constants for processing
SAMPLE_RATE = 16000
FRAMES_PER_SECOND = 25
FRAMES_SIZE = SAMPLE_RATE // FRAMES_PER_SECOND  # 640 samples per frame

def process_audio_stream(stream, callback):
    global hubert_model
    res_lst = []
    num_iter = 0

    for chunk in stream:
        # Ensure the chunk is the correct size for a frame
        if len(chunk) < FRAMES_SIZE:
            # Pad the chunk if it's smaller than a frame
            chunk = np.pad(chunk, (0, FRAMES_SIZE - len(chunk)), mode='constant')
        elif len(chunk) > FRAMES_SIZE:
            # Trim the chunk if it's larger than a frame
            chunk = chunk[:FRAMES_SIZE]

        # Convert chunk to input values
        input_values = wav2vec2_processor(chunk, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
        input_values = input_values.to(device)

        # Process the chunk
        hidden_states = hubert_model(input_values).last_hidden_state
        res_lst.append(hidden_states[0])

    # Concatenate the results
    ret = torch.cat(res_lst, dim=0).cpu()
    if callback is not None:
        callback(ret)
    return ret
