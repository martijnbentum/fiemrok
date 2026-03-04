import pandas as pd
import string

def load_excel():
    df = pd.read_excel('trialsNewTryWithReps.ods')
    return df


def get_experiment_data(df = None):
    if df is None: df = load_excel()
    header = list(df.columns)
    data = [list(line) for line in df.values]
    return header, data

class Experiment:
    def __init__(self, header = None, data = None):
        if header is None or data is None:
            header, data = get_experiment_data()
        self.header = header
        self.data = data
        self.create_trials_and_stimuli()

    def create_trials_and_stimuli(self):
        self.trials = [Trial(line, self) for line in self.data]
        self.stimuli = []
        self.bad_stimuli = []
        for trial in self.trials:
            for stimulus in trial.stimuli:
                self.stimuli.append(stimulus)
            for stimulus in trial.bad_stimuli:
                self.bad_stimuli.append(stimulus)
        self.targets = [x for x in self.stimuli if x.target and x.word_type == 'word']
        self.fillers = [x for x in self.stimuli if x.filler and x.word_type == 'word']
        self.initial_targets = [x for x in self.targets if x.position == 'Initial']
        self.final_targets = [x for x in self.targets if x.position == 'Final']
        self.target_words = sorted(list(set([x.trial.word for x in self.targets])))
        self.final_target_words = sorted(list(set([x.trial.word for x in self.final_targets])))
        self.initial_target_words = sorted(list(set([x.trial.word for x in self.initial_targets])))
        self.filler_words = sorted(list(set([x.trial.word for x in self.fillers])))

    

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
        self.fillter = not self.target


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
        for h,v,c in zip(self.experiment.header, self.line, string.ascii_uppercase):
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

    def __repr__(self):
        m = f'Stimulus({self.trial.word} {self.word_type}' 
        m += f' {self.aligned_type} {self.position})'
        m = f'{m:<44} disc: {self.disc:<12}'
        m += f' audio: {self.audio_filename}'
        return m
    
    def _set_info(self):
        self.audio_filename = f'{self.trial.word}'
        self.disc_word= ''
        self.disc_context = ''
        self.ok = True
        if self.word_type == 'word': 
            self.audio_filename += 'D'
            self.disc_word += self.trial.disc_word
        elif self.word_type == 'non-word': 
            self.audio_filename += 'N'
            self.disc_word += self.trial.disc_non_word
        else: 
            m = f'word_type should be word or non-word, not {word_type}'
            raise ValueError(m)
        if self.aligned_type == 'aligned': 
            self.audio_filename += 'A'
            try:self.disc_context += self.trial.disc_context
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
            m = f'aligned_type should be aligned / misaligned, not {aligned_type}'
            raise ValueError(m)
        self.audio_filename += '.wav'
        if self.position == 'Initial':
            self.disc = self.disc_word + self.disc_context
        elif self.position == 'Final':
            self.disc = self.disc_context + self.disc_word
        self.textgrid_filename = self.audio_filename.replace('.wav', '.tim')
    
        
            


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
