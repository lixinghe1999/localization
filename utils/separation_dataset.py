import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import os

class FUSSDataset(Dataset):
    """Dataset class for FUSS [1] tasks.

    Args:
        file_list_path (str): Path to the txt (csv) file created at stage 2
            of the recipe.
        return_bg (bool): Whether to return the background along the mixture
            and sources (useful for SIR, SAR computation). Default: False.

    References
        [1] Scott Wisdom et al. "What's All the FUSS About Free Universal
        Sound Separation Data?", 2020, in preparation.
    """

    dataset_name = "FUSS"

    def __init__(self, file_list_path, n_src=2, duration=10, sample_rate=16000, return_bg=False):
        super().__init__()
        # Arguments
        dataset_path = os.path.dirname(file_list_path) + '/'
        self.dataset_path = dataset_path
        self.return_bg = return_bg
        self.return_frames = return_frames
        self.return_clap = return_clap
        # Constants
        self.max_n_fg = 3
        self.n_src = n_src  # Same variable as in WHAM
        self.sample_rate = sample_rate
        self.num_samples = self.sample_rate * duration

        # Load the file list as a dataframe
        # FUSS has a maximum of 3 foregrounds, make column names
        self.fg_names = [f"fg{i}" for i in range(self.max_n_fg)]
        names = ["mix", "bg"] + self.fg_names
        # Lines with less labels will have nan, replace with empty string
        self.mix_df = pd.read_csv(file_list_path, sep="\t", names=names)
        # Number of foregrounds (fg) vary from 0 to 3
        # This can easily be used to remove mixtures with less than x fg
        remove_less_than = n_src + 2
        self.mix_df.dropna(thresh=remove_less_than, inplace=True)
        self.mix_df.reset_index(inplace=True)
        self.mix_df.fillna(value="", inplace=True)

    def __len__(self):
        return len(self.mix_df)

    def __getitem__(self, idx):
        # Each line has absolute to miture, background and foregrounds
        line = self.mix_df.iloc[idx]
        # mix = librosa.load(self.dataset_path + line["mix"], sr=self.sample_rate)[0][:self.num_samples]
        sources = []
        num_sources = 0
        for fg_path in [line[fg_n] for fg_n in self.fg_names]:
            if fg_path:
                source = librosa.load(self.dataset_path + fg_path, sr=self.sample_rate)[0]
                num_sources += 1
            else:
                source = np.zeros(self.num_samples, dtype=np.float32)
            source = source[:self.num_samples]
            sources.append(source)
        sources = sources[:self.n_src]
        mix = np.sum(sources, axis=0).astype(np.float32)
        sources = torch.from_numpy(np.vstack(sources))

        if self.return_bg:
            bg = sf.read(self.dataset_path + line["bg"], dtype="float32")[0]
            mix += torch.from_numpy(bg[:self.num_samples])
        return torch.from_numpy(mix), sources
    

    def get_infos(self):
        """Get dataset infos (for publishing models).

        Returns:
            dict, dataset infos with keys `dataset`, `task` and `licences`.
        """
        infos = dict()
        infos["dataset"] = self.dataset_name
        infos["task"] = "sep_noisy"
        infos["licenses"] = [fuss_license]
        return infos


fuss_license = dict(
    title="Free Universal Sound Separation Dataset",
    title_link="https://zenodo.org/record/3743844#.X0Jtehl8Jkg",
    author="Scott Wisdom; Hakan Erdogan; Dan Ellis and John R. Hershey",
    author_link="https://scholar.google.com/citations?user=kJM6N7IAAAAJ&hl=en",
    license="Creative Commons Attribution 4.0 International",
    license_link="https://creativecommons.org/licenses/by/4.0/legalcode",
    non_commercial=False,
)

if __name__ == '__main__':
    import torchmetrics
    import matplotlib.pyplot as plt
    metric = torchmetrics.ScaleInvariantSignalNoiseRatio()
    dataset = FUSSDataset('dataset/FUSS/ssdata/', 'dataset/FUSS/ssdata/eval_example_list.txt', n_src=2, duration=10, sample_rate=8000)
    print(len(dataset))
    for data in dataset:
        # print(data[0].shape, data[1].shape)
        mix, sources = data
        print(mix.shape, sources.shape)
        sisnr_1 = metric(sources[0], mix)
        sisnr_2 = metric(sources[1], mix)
        print(sisnr_1, sisnr_2)

        fig, axs = plt.subplots(3, 1)
        axs[0].plot(mix)
        axs[0].set_title('mix')
        axs[1].plot(sources[0])
        axs[1].set_title('source 1')
        axs[2].plot(sources[1])
        axs[2].set_title('source 2')

        plt.savefig('test.png')

        break