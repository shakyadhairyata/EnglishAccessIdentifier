import os
import torchaudio
import pandas as pd
from tqdm import tqdm

def load_common_voice(csv_path, audio_dir, target_accents):
    df = pd.read_csv(csv_path)
    df = df[df['accent'].isin(target_accents)]
    data = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = os.path.join(audio_dir, row['path'])
        label = target_accents.index(row['accent'])
        data.append((path, label))
    return data