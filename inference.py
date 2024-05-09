import torch
import torch.nn.functional as F
from SpeechLM import SpeechLMConfig, SpeechLM

model_path = '/home_new/SpeechT5/SpeechLM/models/speechlmh_base_checkpoint_clean.pt'
checkpoint = torch.load(model_path)
cfg = SpeechLMConfig(checkpoint['cfg']['model'])
model = SpeechLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

wav_input_16khz = torch.randn(1,10000)
normalize = checkpoint['cfg']['task']['normalize']  # False for base model, True for large model
if normalize:
    wav_input_16khz = F.layer_norm(wav_input_16khz[0], wav_input_16khz[0].shape).unsqueeze(0)

# extract the representation of last layer
rep = model.extract_features(wav_input_16khz)[0]

# extract the representation of each layer
output_layer = model.cfg.encoder_layers + model.cfg.text_transformer.encoder.layers
rep, layer_results = model.extract_features(wav_input_16khz, output_layer=output_layer, ret_layer_results=True)[0]
layer_reps = [x.transpose(0, 1) for x in layer_results]

print(rep.shape, layer_reps[0].shape, layer_reps[-1].shape)
# torch.Size([1, 31, 768]) torch.Size([1, 31, 768]) torch.Size([1, 31, 768])