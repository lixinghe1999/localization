from torch.utils.data import Dataset
from .feature import gccphat, mel_spec, source_separation, shift_mixture
from .label import Gaussian, filter_label
import os
import librosa
import json
import numpy as np
import pandas as pd
class Separation_dataset(Dataset):
    def __init__(self, dataset, config=None, ):
        self.config = config
        self.root_dir = dataset
        self.sr = self.config['model']['sample_rate']
        self.output_format = self.config['output_format']   
        self.duration = self.config['duration']

        with open(os.path.join(self.root_dir, 'label.json')) as f:
            self.labels = json.load(f)
        self.labels = [label for label in self.labels]
        self.labels = filter_label(self.labels, self.config['max_azimuth'], self.config['min_azimuth'])

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx): 
        label = self.labels[idx]
        file_folder = os.path.join(self.root_dir, label['fname'])
        source_audio = [librosa.load(os.path.join(file_folder, file), sr=self.sr, mono=False)[0] for file in os.listdir(file_folder)]
        source_audio = np.array([prune_extend(audio, self.duration * self.sr) for audio in source_audio])
    
        mixture_audio = np.sum(source_audio, axis=0) # source_audio [source, left+right, T], mixture_audio [left+right, T]

        if self.output_format == 'separation':
            # fix to left channel 
            mixture_audio = mixture_audio[0] # [left, T]
            source_audio = source_audio[:, 0] # [source, left, T]
            return mixture_audio, source_audio
        elif self.output_format == 'beamforming':
            # beamforming separation
            if np.random.uniform() < -1: # point to negative
                # make sure the doa is not close to the source
                doa_candidates = []
                for i in range(0, 360):
                    for doa in label['doa_degree']:
                        if np.abs(doa[0] - i) > 10:
                            doa_candidates.append(i)
                target_doa = np.random.choice(doa_candidates, 1, replace=False)
                target_range = 1
                target_position = np.array([np.cos(np.radians(target_doa)), np.sin(np.radians(target_doa))]) * target_range
                mixture_audio, shift = shift_mixture(mixture_audio, target_position, 0.09, self.sr) # shift to the left channel, [left+right, T]
                source_audio = np.zeros_like(mixture_audio)
            else:
                choose_idx = np.random.choice(range(len(label['doa_degree'])), 1, replace=False)[0]
                target_doa = label['doa_degree'][choose_idx][0]
                target_range = label['range'][choose_idx][0]
                target_position = np.array([np.cos(np.radians(target_doa)), np.sin(np.radians(target_doa))]) * target_range

                mixture_audio, shift = shift_mixture(mixture_audio, target_position, 0.09, self.sr) # shift to the left channel, [left+right, T]
                source_audio, shift = shift_mixture(source_audio[choose_idx], target_position, 0.09, self.sr) # shift to the left channel, [left+right, T]
            return mixture_audio, source_audio
        elif self.output_format == 'region':
            number_of_regions = self.config['model']['num_src']
            regions = np.linspace(self.config['min_azimuth'], self.config['max_azimuth'], number_of_regions + 1)
            
            region_audio = np.zeros((number_of_regions, self.config['model']['sample_rate'] * self.config['duration']), dtype=np.float32)
            region_active = np.zeros(number_of_regions, dtype=np.int32)
            for doa, range, _source_audio in zip(label['doa_degree'], label['range'], source_audio):
                region_idx = np.digitize(doa[0], regions) - 1
                region_active[region_idx] = 1
                region_audio[region_idx] += _source_audio[0] # left channel
            return mixture_audio, region_audio # [left+right, T], [region, left, T]
       
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

    def __init__(self, file_list_path, sample_rate, return_bg=False):
        super().__init__()
        # Arguments
        self.folder = os.path.dirname(file_list_path)
        self.return_bg = return_bg
        # Constants
        self.max_n_fg = 2
        self.n_src = self.max_n_fg  # Same variable as in WHAM
        self.sample_rate = sample_rate
        self.num_samples = self.sample_rate * 10

        # Load the file list as a dataframe
        # FUSS has a maximum of 3 foregrounds, make column names
        self.fg_names = [f"fg{i}" for i in range(self.max_n_fg)]
        names = ["mix", "bg"] + self.fg_names
        # Lines with less labels will have nan, replace with empty string
        self.mix_df = pd.read_csv(file_list_path, sep="\t", names=names)
        # Number of foregrounds (fg) vary from 0 to 3
        # This can easily be used to remove mixtures with less than x fg
        # remove_less_than = 2
        # self.mix_df.dropna(thresh=remove_less_than, inplace=True)
        # self.mix_df.reset_index(inplace=True)

        # only keep the mixtures where there are exactly 2 sources
        self.mix_df = self.mix_df[self.mix_df[self.fg_names].notna().sum(axis=1) == 2]
        self.mix_df.reset_index(inplace=True, drop=True)

        self.mix_df.fillna(value="", inplace=True)

    def __len__(self):
        return len(self.mix_df)

    def __getitem__(self, idx):
        # Each line has absolute to miture, background and foregrounds
        line = self.mix_df.iloc[idx]
        mix = librosa.load(os.path.join(self.folder, line["mix"]), sr=self.sample_rate, mono=True)[0]
        sources = []
        num_sources = 0
        for fg_path in [line[fg_n] for fg_n in self.fg_names]:
            if fg_path:
                num_sources += 1
                source = librosa.load(os.path.join(self.folder, fg_path), sr=self.sample_rate, mono=True)[0]
            else:
                source = np.zeros_like(mix)
            sources.append(source)
        # print('num_sources:', num_sources)
        sources = torch.from_numpy(np.vstack(sources))

        if self.return_bg:
            bg = librosa.load(os.path.join(self.folder, line["bg"]), sr=self.sample_rate, mono=True)[0]
            return torch.from_numpy(mix), sources, torch.from_numpy(bg)
        # print('mix shape:', mix.shape, 'sources shape:', sources.shape)
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