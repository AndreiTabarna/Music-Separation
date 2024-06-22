
#Here, the data augmentation tool is implemented

import scaper
import nussl
import nussl.datasets.transforms as nussl_tfm
import torch
from pathlib import Path
import tqdm
import sys
import numpy as np
import warnings
from typing import Union, List
import logging
import os
from . import utils

MAX_SOURCE_TIME = 10000
LABELS = ['bass', 'drums', 'other', 'vocals']

def mixer(
    stft_params,
    transform,
    num_mixtures : int = 10,
    fg_path : str = 'data/train',
    duration : float = 5.0,
    sample_rate : int = 44100,
    ref_db : Union[float, List] = [-30, -10],
    n_channels : int = 1,
    master_label : str = 'vocals',
    source_file : List = ['choose', []],
    snr : List = ('uniform', -5, 5),
    target_instrument : str = 'vocals',
    target_snr_boost : float = 0.0,
    pitch_shift : List = ('uniform', -2, 2),
    time_stretch : List = ('uniform', 0.9, 1.1),
    coherent_prob : float = 0.5,
    augment_prob : float = 0.5,
    quick_pitch_time_prob : float = 1.0,
    overfit : bool = False,
    overfit_seed : int = 0,
):
    mix_closure = MUSDBMixer(
        fg_path, duration, sample_rate, ref_db, n_channels, 
        master_label, source_file, snr, pitch_shift, time_stretch,
        coherent_prob, augment_prob, quick_pitch_time_prob,
        overfit, overfit_seed, target_instrument, target_snr_boost,
    )
    dataset = nussl.datasets.OnTheFly(
        mix_closure, num_mixtures, stft_params=stft_params,
        transform=transform, sample_rate=sample_rate
    )
    return dataset

on_the_fly = mixer

class MUSDBMixer():
    def __init__(
        self,
        fg_path : str,
        duration : float,
        sample_rate : int,
        ref_db : Union[float, tuple],
        n_channels : int = 1,
        master_label : str = 'vocals',
        # Event parameters
        source_file=('choose', []),
        snr=('uniform', -5, 5),
        pitch_shift=('uniform', -2, 2),
        time_stretch=('uniform', 0.9, 1.1),
        # Generation parameters
        coherent_prob=0.5,
        augment_prob=0.5,
        quick_pitch_time_prob=1.0,
        overfit=False,
        overfit_seed=0,
        target_instrument='vocals',
        target_snr_boost=0,
    ):
        pitch_shift = (
            tuple(pitch_shift) 
            if pitch_shift 
            else None
        )
        time_stretch = (
            tuple(time_stretch) 
            if time_stretch 
            else None
        )
        snr = tuple(snr)
        self.base_event_parameters = {
            'label': ('const', master_label),
            'source_file': ('choose', []),
            'source_time': ('uniform', 0, MAX_SOURCE_TIME),
            'event_time': ('const', 0),
            'event_duration': ('const', duration),
            'snr': snr,
            'pitch_shift': pitch_shift,
            'time_stretch': time_stretch
        }
        self.fg_path = fg_path
        self.sample_rate = sample_rate
        self.ref_db = ref_db
        self.n_channels = n_channels
        self.duration = duration
        self.coherent_prob = coherent_prob
        self.augment_prob = augment_prob
        self.quick_pitch_time_prob = quick_pitch_time_prob

        self.overfit = overfit
        self.overfit_seed = overfit_seed
        
        self.target_instrument = target_instrument
        self.target_snr_boost = target_snr_boost

    def _create_scaper_object(self, state):
        sc = scaper.Scaper(
            self.duration, self.fg_path, self.fg_path,
            random_state=state
        )
        sc.sr = self.sample_rate
        sc.n_channels = self.n_channels
        ref_db = self.ref_db
        if isinstance(ref_db, List):
            ref_db = state.uniform(ref_db[0], ref_db[1])
        sc.ref_db = ref_db
        return sc

    def _add_events(self, sc, event_parameters, event=None):
        labels = ['vocals', 'drums', 'bass', 'other']
        snr_dist = event_parameters.pop('snr')
        for label in labels:
            _snr_dist = list(snr_dist).copy()
            if label == self.target_instrument:
                _snr_dist[1] += self.target_snr_boost
                _snr_dist[2] += self.target_snr_boost
            event_parameters['label'] = ('const', label)
            if event:
                event_parameters['source_file'] = (
                    'const', event.source_file.replace('vocals', label)
                )
            sc.add_event(snr=tuple(_snr_dist), **event_parameters)

    def incoherent(self, sc):
        event_parameters = self.base_event_parameters.copy()
        if sc.random_state.rand() > self.augment_prob:
            event_parameters['pitch_shift'] = None
            event_parameters['time_stretch'] = None

        self._add_events(sc, event_parameters)
        quick_pitch_time = sc.random_state.rand() <= self.quick_pitch_time_prob
        return sc.generate(fix_clipping=True, quick_pitch_time=quick_pitch_time)

    def coherent(self, sc):
        event_parameters = self.base_event_parameters.copy()
        if sc.random_state.rand() > self.augment_prob:
            event_parameters['pitch_shift'] = None
            event_parameters['time_stretch'] = None

        sc.add_event(**event_parameters)
        event = sc._instantiate_event(sc.fg_spec[0])
        sc.reset_fg_event_spec()
        
        event_parameters['source_time'] = ('const', event.source_time)
        if event_parameters['pitch_shift'] is not None:
            event_parameters['pitch_shift'] = ('const', event.pitch_shift)
        if event_parameters['time_stretch'] is not None:
            event_parameters['time_stretch'] = ('const', event.time_stretch)

        self._add_events(sc, event_parameters, event)
        quick_pitch_time = sc.random_state.rand() <= self.quick_pitch_time_prob
        return sc.generate(fix_clipping=True, quick_pitch_time=quick_pitch_time)
    
    def __call__(self, dataset, i):
        if self.overfit:
            i = self.overfit_seed
        state = np.random.RandomState(i)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            sc = self._create_scaper_object(state)
            if state.rand() < self.coherent_prob:
                data = self.coherent(sc)
            else:
                data = self.incoherent(sc)
        
        soundscape_audio, soundscape_jam, annotation_list, event_audio_list = data
        
        mix = dataset._load_audio_from_array(
            audio_data=soundscape_audio, sample_rate=dataset.sample_rate
        )
        sources = {}
        ann = soundscape_jam.annotations.search(namespace='scaper')[0]
        for obs, event_audio in zip(ann.data, event_audio_list):
            key = obs.value['label']
            sources[key] = dataset._load_audio_from_array(
                audio_data=event_audio, sample_rate=dataset.sample_rate
            )
        
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': {
                'jam': soundscape_jam,
                'idx': i
            }
        }
        return output
