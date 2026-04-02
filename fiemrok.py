import string

import pandas as pd

import annotation
import audio
import locations
import ssh_audio_play


def load_excel(filename = locations.excel_filename):
    df = pd.read_excel(filename)
    return df


def get_experiment_data(df = None):
    if df is None: df = load_excel()
    header = list(df.columns)
    data = [list(line) for line in df.values]
    return header, data


class Experiment:
    def __init__(self, header = None, data = None, audio_info_dict = None):
        if header is None or data is None:
            header, data = get_experiment_data()
            if audio_info_dict is None:
                audio_info_dict = audio.make_or_load_audio_info_dict()
        if audio_info_dict is None: audio_info_dict = {}
        self.audio_info_dict = audio_info_dict
        self.header = header
        self.data = data
        self._create_trials_and_stimuli()
        self._set_info()

    def __repr__(self):
        return f'Experiment with {len(self.final_targets)} stimuli'

    def _create_trials_and_stimuli(self):
        self.trials = [Trial(line, self) for line in self.data]
        self.stimuli = []
        self.bad_stimuli = []
        for trial in self.trials:
            for stimulus in trial.stimuli:
                self.stimuli.append(stimulus)
            for stimulus in trial.bad_stimuli:
                self.bad_stimuli.append(stimulus)

    def _set_info(self):
        targets = [x for x in self.stimuli if x.target and
            x.word_type == 'word']
        self.targets = targets
        fillers = [x for x in self.stimuli if x.filler and
            x.word_type == 'word']
        self.fillers = fillers
        initial_targets = [x for x in self.targets if x.position == 'Initial']
        self.initial_targets = initial_targets
        self.final_targets = [x for x in self.targets if x.position == 'Final']

        target_words = sorted(list(set([x.trial.word for x in self.targets])))
        self.target_words = target_words
        ftw = sorted(list(set([x.trial.word for x in self.final_targets])))
        self.final_target_words = ftw
        itw = sorted(list(set([x.trial.word for x in self.initial_targets])))
        self.initial_target_words = itw
        filler_words = sorted(list(set([x.trial.word for x in self.fillers])))
        self.filler_words = filler_words
 

class Trial:
    def __init__(self, line, experiment):
        self.line = line
        self.experiment = experiment
        self._set_info()
        self._make_stimuli()

    def _set_info(self):
        self.disc_word = self.line[0]
        self.frequency = self.line[1]
        self.condition = self.line[2]
        self.word = self.line[3]
        self.disc_context = self.line[4]
        self.disc_constraint_1 = self.line[5]
        self.disc_alternation = self.line[6]
        self.disc_constraint_2 = self.line[7]
        self.disc_non_word = self.line[8]
        self.disc_target_phoneme = self.line[9]
        self.position = self.line[13]
        self.target = 'F' not in self.condition
        self.filler = not self.target
        self.fillter = self.filler

    def __repr__(self):
        return f'Trial({self.word} {self.condition} {self.position})'

    def __str__(self):
        m = f'word: {self.word}\n'
        m += f'condition: {self.condition}\n'
        m += f'position: {self.position}\n'
        m += f'disc word: {self.disc_word}\n'
        m += f'disc context: {self.disc_context}\n'
        m += f'constraint 1: {self.disc_constraint_1}\n'
        m += f'disc alternation: {self.disc_alternation}\n'
        m += f'constraint 2: {self.disc_constraint_2}\n'
        m += f'disc non-word: {self.disc_non_word}\n'
        m += f'disc target phoneme: {self.disc_target_phoneme}\n'
        m += f'position: {self.position}\n'
        return m

    def print_line(self):
        for h, v, c in zip(self.experiment.header, self.line,
            string.ascii_uppercase):
            print(f'{c}:   {h[:10]}:   {v}')

    def _make_stimuli(self):
        self.stimuli = []
        self.bad_stimuli = []
        for word_type in ['word', 'non-word']:
            for aligned_type in ['aligned', 'misaligned']:
                s = Stimulus(self, word_type, aligned_type)
                if s.ok: self.stimuli.append(s)
                else: self.bad_stimuli.append(s)

class Stimulus():
    def __init__(self, trial, word_type = 'word', aligned_type = 'aligned'):
        self.trial = trial
        self.word_type = word_type
        self.aligned_type = aligned_type
        self.experiment = trial.experiment
        self.position = self.trial.position
        self.target = self.trial.target
        self.filler = not self.target
        self._set_info()
        self._get_audio_info()
        self._make_segments()

    def __repr__(self):
        m = f'Stimulus({self.trial.word} {self.word_type}'
        m += f' {self.aligned_type} {self.position})'
        m = f'{m:<44} disc: {self.disc:<12}'
        m += f' audio: {self.audio_filename}'
        return m
    
    def _set_info(self):
        self.audio_filename = f'{self.trial.word}'
        self.disc_word = ''
        self.disc_context = ''
        self.ok = True
        if self.word_type == 'word':
            self.audio_filename += 'W'
            self.disc_word += self.trial.disc_word
        elif self.word_type == 'non-word':
            self.audio_filename += 'N'
            self.disc_word += self.trial.disc_non_word
        else:
            m = f'word_type should be word or non-word, not {self.word_type}'
            raise ValueError(m)
        if self.aligned_type == 'aligned':
            self.audio_filename += 'A'
            try: self.disc_context += self.trial.disc_context
            except TypeError:
                self.ok = False
                pass
        elif self.aligned_type == 'misaligned':
            self.audio_filename += 'M'
            try: self.disc_context += self.trial.disc_alternation
            except TypeError:
                self.ok = False
                pass
        else:
            m = f'aligned_type should be aligned / misaligned, not '
            m += f'{self.aligned_type}'
            raise ValueError(m)
        self.audio_filename += '.wav'
        if self.position == 'Initial':
            self.disc = self.disc_word + self.disc_context
        elif self.position == 'Final':
            self.disc = self.disc_context + self.disc_word
        self.textgrid_filename = self.audio_filename.replace('.wav', '.tim')

    def _get_audio_info(self):
        try: d = self.trial.experiment.audio_info_dict[self.audio_filename]
        except KeyError:
            self.ok = False
            return
        self.audio_info = d
        self.audio_path = d['filename']
        self.audio_duration = self.audio_info['duration']

    def _make_segments(self):
        if not self.ok or not self.audio_duration: return
        p = locations.textgrid_filename_to_path(self.textgrid_filename)
        self.textgrid_path = p
        self.textgrid = annotation.load_textgrid(p)
        self.time_points = annotation.extract_time_points(self.textgrid)
        self._make_context_segment()
        self._make_target_segment()
        self._make_cluster_segment()
        self.segments = [self.target_segment, self.context_segment,
            self.cluster_segment]

    def _make_context_segment(self):
        if self.position == 'Final':
            start = 0.0            
            end = self.time_points[1]
        if self.position == 'Initial':
            start = self.time_points[2]
            end = self.audio_duration
        self.context_segment = Segment(start, end, self.disc_context, 
            self, 'context')

    def _make_target_segment(self):
        if self.position == 'Final':
            start = self.time_points[1]
            end = self.audio_duration
        if self.position == 'Initial':
            start = 0.0
            end = self.time_points[1]
        self.target_segment = Segment(start, end, self.disc_word, 
            self, 'target')
    
    def _make_cluster_segment(self):
        start = self.time_points[0]
        end = self.time_points[2]
        if self.position == 'Final':
            self.disc_cluster = self.disc_context[-1] + self.disc_word[0]
        elif self.position == 'Initial':
            self.disc_cluster = self.disc_word[-1] + self.disc_context[0]
        self.cluster_segment = Segment(start, end, self.disc_cluster, 
            self, 'cluster')

    @property
    def play(self):
        print(f'Playing {self.audio_filename}, {self.disc}')
        ssh_audio_play.play_audio(self.audio_path)


class Segment:
    def __init__(self, start, end, label, stimulus, segment_type = None):
        self.start = start
        self.end = end
        self.label = label
        self.stimulus = stimulus
        self.segment_type = segment_type
        self.audio_path = stimulus.audio_path
    
    def __repr__(self):
        m = f'Segment({self.start:.2f}, {self.end:.2f}, {self.label})'
        if self.segment_type is not None:
            m += f' type: {self.segment_type}'
        return m

    @property
    def play(self):
        m = f'Playing {self.audio_path}, {self.label}'
        m += f' {self.stimulus.disc} {self.stimulus.trial.word}'
        print(m)
        ssh_audio_play.play_audio(self.audio_path, start = self.start,
            end = self.end)
        
'''
{ort_woord(D)}{W=woord(D),N=non-woord(I)}{A=aligned(E),M=misaligned(G)}{}.wav

kolom N geeft aan of woord (kolom d) initieel is of final

alignments van de stimuli in .tim bestanden

Bij initial woorden
1 geeft einde van het eerste deel aan
2 geeft einde van de eerste foon van de onset van het tweede deel
3 geeft einde van de tweede foon van de onset van het tweede deel

Bij final woorden
tot 1 totaan het cluster
en na 1 tot 2 eerste foon van het cluster
na 2 target woord

2 start target woord

3 is einde cluster
'''
