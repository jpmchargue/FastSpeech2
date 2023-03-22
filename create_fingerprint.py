import argparse
import librosa
import math
import numpy as np
import os
import torch

import fingerprint as Fingerprint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    type=str,
    default=None,
    help="the path to a wav file to generate a fingerprint for",
)
parser.add_argument(
    "--saveto",
    type=str,
    default=None,
    help="the path to save the fingerprint to",
)

args = parser.parse_args()
cwd = os.getcwd()
verifier = Fingerprint.model.Verifier()
print("Loading model...")
checkpoint_name = "fingerprint/verifier_state35000.pt"
if os.path.exists(os.path.join(cwd, checkpoint_name)):
    verifier.load_state_dict(torch.load(checkpoint_name))

y, sr = librosa.load(args.source)
S = (np.transpose(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=math.ceil(sr*0.025), hop_length=math.ceil(sr*0.01), n_mels=40), ref=np.max)))
fingerprint = verifier.get_embedding(S)

np.save(args.saveto, fingerprint)