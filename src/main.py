
from midi_loader import MIDILoader
from pathlib import Path
from midi_processors import FingerdexProcessor

"""locate data folder with the midi recordings"""
SCRIPT_DIR = Path(__file__).resolve().parent
print(SCRIPT_DIR)
root_folder = (SCRIPT_DIR / "../data/midi_recordings").resolve()


"""load the data from the midi files"""
loader = MIDILoader()
data_fingertest, data_states = loader.load_midi(root_folder=root_folder)

"""process the data fingertest"""
finger_processor = FingerdexProcessor()
print(finger_processor.analyse_fingertest(data_fingertest))


