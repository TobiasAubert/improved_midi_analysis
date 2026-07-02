from midi_loader import MIDILoader
from pathlib import Path
import pandas as pd
from midi_processors import FingerdexProcessor, StatesProcessor
from data_analyser import FingerDataAnalyser, StateDataAnalyser
import matplotlib.pyplot as plt
from data_plotter import FingerDataPlotter, StateDataPlotter


"""locate data folder with the midi recordings"""
SCRIPT_DIR = Path(__file__).resolve().parent
root_folder = (SCRIPT_DIR / "../data/midi_recordings").resolve()
plot_dir = (SCRIPT_DIR / "../plots").resolve()
statistical_test_dir = (SCRIPT_DIR / "../statistical_tests").resolve()
fingerdex_report_dir = statistical_test_dir / "fingerdex"
states_report_dir = statistical_test_dir / "states"


"""load the data from the midi files"""
loader = MIDILoader()
data_fingertest, data_states = loader.load_midi(root_folder=root_folder)

"""process the data fingertest"""
finger_processor = FingerdexProcessor()
df_fingertest = finger_processor.process_fingerdata(data_fingertest)
df_fingertest = finger_processor.fix_wrong_seq(df_fingertest)
df_fingertest = df_fingertest[df_fingertest["keystrokes"] > 10] # some recordings have less than 10 key strokes and that is anusual
df_fingertest[["participant_id", "test", "correct_sequences", "most_common_sequence"]].to_csv("fingerdex.csv", index=False)


"""procces the data with states"""
state_processor = StatesProcessor()
df_states = state_processor.process_statedata(data_states)
df_states = state_processor.calculate_transitions(df_states)

#Remove invalid data
df_states = df_states[df_states["notes"].apply(bool)] # removes entreis with no played key
df_states = state_processor.remove_incomplete_seq(df_states, show_removed=True) #removes all entreis in which not enough pitches are played to have played all states (the whole seq)

state_processor.check_state_count(df_states)

"""Statistical analysis"""
fa = FingerDataAnalyser(fingerdex_report_dir)
report_paths = fa.run_analysis_pipeline(df_fingertest)
sa = StateDataAnalyser(states_report_dir)
state_report_paths = sa.analyze_transition_pipeline(df_states)


"""Plotting"""
dp = FingerDataPlotter(plot_dir)
dp.boxplot(df_fingertest)

pl = StateDataPlotter(plot_dir)
pl.learning_curve(df_states)
pl.pre_post_boxplot(df_states)
pl.pre_post_violin(df_states)
