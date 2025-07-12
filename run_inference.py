#!/usr/bin/env python3

import sys
import os
import torch
import torchaudio
from pathlib import Path
from lhotse import RecordingSet
from lhotse import Recording
from lhotse.features.kaldi import Fbank, FbankConfig
from lhotse.audio import Recording
import soundfile as sf
import tempfile
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Add local module paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pruned_transducer_stateless7"))

from pruned_transducer_stateless7.model import Transducer
from pruned_transducer_stateless7.decoder import Decoder
from pruned_transducer_stateless7.joiner import Joiner
from pruned_transducer_stateless7.zipformer import Zipformer
from inference_params import DummyParams
from pruned_transducer_stateless7.decode import greedy_search
def to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))

def build_model(params):
    encoder = Zipformer(
        num_features=params.feature_dim,
        output_downsampling_factor=2,
        zipformer_downsampling_factors=to_int_tuple(params.zipformer_downsampling_factors),
        encoder_dims=to_int_tuple(params.encoder_dims),
        attention_dim=to_int_tuple(params.attention_dims),
        encoder_unmasked_dims=to_int_tuple(params.encoder_unmasked_dims),
        nhead=to_int_tuple(params.nhead),
        feedforward_dim=to_int_tuple(params.feedforward_dims),
        cnn_module_kernels=to_int_tuple(params.cnn_module_kernels),
        num_encoder_layers=to_int_tuple(params.num_encoder_layers),
    )

    decoder = Decoder(
        vocab_size=params.vocab_size,
        decoder_dim=params.decoder_dim,
        blank_id=params.blank_id,
        context_size=params.context_size,
    )

    joiner = Joiner(
        encoder_dim=int(params.encoder_dims.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )

    return Transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        encoder_dim=int(params.encoder_dims.split(",")[-1]),
        decoder_dim=params.decoder_dim,
        joiner_dim=params.joiner_dim,
        vocab_size=params.vocab_size,
    )

def load_model():
    exp_dir = "pruned_transducer_stateless7/exp_fresh"
    checkpoint = "epoch-70.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = DummyParams()
    model = build_model(params)
    model.load_state_dict(torch.load(f"{exp_dir}/{checkpoint}", map_location=device)["model"])
    model.eval().to(device)

    return model, device

def preprocess_audio(wav_path):
    from lhotse.features.kaldi import Fbank, FbankConfig

    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    print(f"Original sample rate: {sample_rate}")
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate} to 16000")
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    else:
        print("No resampling needed")
    print(f"Waveform shape after loading: {waveform.shape}")
     # Take the first channel (if stereo)
    audio_samples = waveform.squeeze().numpy()

    # Create the Fbank extractor
    fbank = Fbank(FbankConfig(num_mel_bins=80))

    # Extract features directly from samples (as np.ndarray) + sampling_rate
    features = fbank.extract(audio_samples, sampling_rate=sample_rate)

    # Convert to torch tensor and return
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)



def load_tokens():
    token_path = "data/hu/lang_bpe_500/tokens.txt"
    with open(token_path, "r") as f:
        return [line.strip() for line in f]

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_inference_wav.py path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    print(f"📂 Processing: {wav_path}")

    model, device = load_model()
    features = preprocess_audio(wav_path).to(device)
    id2token = load_tokens()
    with torch.no_grad():
        x_lens = torch.tensor([features.size(1)], dtype=torch.int32).to(device)
        encoder_out, encoder_out_lens = model.encoder(features, x_lens)
        hyp = greedy_search(model, encoder_out, encoder_out_lens)
        tokens = hyp
    print("Raw hyp:", hyp)
    print("Decoded tokens:", tokens)
    print("Token indices:", [t for t in tokens])
    print("Length of tokens:", len(tokens))
    print("Encoder out shape:", encoder_out.shape)
    print("Encoder lens:", encoder_out_lens)
    if isinstance(tokens, (int, torch.Tensor)):
        tokens = [tokens]  # Convert single token to list
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load("data/hu/lang_bpe_500/bpe.model")

    text = sp.decode(tokens)

    print("\n🔊 Transcription:")
    print(text)

if __name__ == "__main__":
    main()
