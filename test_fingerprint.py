import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import torch
import librosa
import math
import umap

import fingerprint as Fingerprint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cwd = os.getcwd()

NUM_SPEAKERS_PLOTTED = 10
NUM_UTTERANCES = 10
DATA_PATH = "raw_data/VCTK"

verifier = Fingerprint.model.Verifier()
print("Loading model...")
checkpoint_name = "fingerprint/verifier_state35000.pt"
if os.path.exists(os.path.join(cwd, checkpoint_name)):
    verifier.load_state_dict(torch.load(checkpoint_name))

print("Loading random sample...")
speakers = random.sample(os.listdir(DATA_PATH), NUM_SPEAKERS_PLOTTED)
all_spectrograms = []
for speaker in speakers:
    print(speaker)
    names = random.sample([name for name in os.listdir(os.path.join(DATA_PATH, speaker)) if name.endswith("wav")], NUM_UTTERANCES)
    paths = [os.path.join(DATA_PATH, speaker, name) for name in names]
    for p in paths:
        print(p)
    spectrograms = []
    for path in paths:
        y, sr = librosa.load(path)
        S = (np.transpose(librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=math.ceil(sr*0.025), hop_length=math.ceil(sr*0.01), n_mels=40), ref=np.max)))
        #plt.imshow(S)
        #plt.show()
        spectrograms.append(S)
    all_spectrograms.extend(spectrograms)

print("Embedding...")
all_embeddings = torch.stack([verifier.get_embedding(spectrogram) for spectrogram in all_spectrograms])

print("Got embeddings!")
for embedding in all_embeddings:
    print(all_embeddings)

print("Mapping...")
scaled = StandardScaler().fit_transform(all_embeddings)
umapped = umap.UMAP().fit_transform(scaled)

print("Plotting...")
colors = ['r', "orange", 'y', 'g', "aqua", 'b', "indigo", 'm', "slategray", "black"]
labels = np.repeat(colors, 10)
mp = [mpatches.Patch(color=c, label='Speaker ' + speakers[i]) for i, c in enumerate(colors)]
plt.scatter(
    umapped[:, 0],
    umapped[:, 1],
    c=labels)
plt.legend(handles=mp)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP Projection of Speaker Embeddings (35000)')
plt.show()