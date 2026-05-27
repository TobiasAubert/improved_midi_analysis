from midi_loader import MIDILoader
from pathlib import Path
import pandas as pd
from midi_processors import FingerdexProcessor, StatesProcessor
from data_analyser import FingerDataAnalyser
import matplotlib.pyplot as plt
from data_plotter import FingerDataPlotter, StateDataPlotter

"""locate data folder with the midi recordings"""
SCRIPT_DIR = Path(__file__).resolve().parent
print(SCRIPT_DIR)
root_folder = (SCRIPT_DIR / "../data/midi_recordings").resolve()


"""load the data from the midi files"""
loader = MIDILoader()
data_fingertest, data_states = loader.load_midi(root_folder=root_folder)

"""process the data fingertest"""
finger_processor = FingerdexProcessor()
df_fingertest = finger_processor.process_fingerdata(data_fingertest)


"""procces the data with states"""
state_processor = StatesProcessor()
df_states = state_processor.process_statedata(data_states)
df_states = state_processor.calculate_transitions(df_states)


"""Plotting"""
dp = FingerDataPlotter()
dp.boxplot(df_fingertest)

pl = StateDataPlotter()
pl.learning_curve(df_states)
pl.pre_post_boxplot(df_states)
pl.pre_post_violin(df_states)

# df_states.to_csv('safe_check_order.csv', index=False, encoding='utf-8-sig')
