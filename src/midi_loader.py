import os
import pretty_midi
import pandas as pd
from pathlib import Path


# Change this to your actual folder path or copy data in this folder

# output_dir = (SCRIPT_DIR / "../../data").resolve()

class MIDILoader:
    def __init__(self):
        pass
        
    def extract_midi_info(self, filename):
        # Extract participant ID, test_name
        participant_id = filename.split('_')[0]
        test = filename.split('_')[1].split('.')[0]
        return(participant_id,test)

    def extract_played_notes(self, midi, exclude_drums=True):
        played_notes = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
                for note in sorted_notes:
                    name = pretty_midi.note_number_to_name(note.pitch)
                    played_notes.append({
                        'pitch': note.pitch,
                        'name': name,
                        'start': note.start,
                        'end': note.end
                })
        return played_notes

    def load_midi(self, root_folder):
        data_fingertest = []
        data_states = []
        # Walk through all subfolders
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith(('.mid', '.midi')):
                    file_path = os.path.join(dirpath, filename)

                    try:
                        participant_id, test = self.extract_midi_info(filename=filename)

                        # Extract note keystrokes
                        midi = pretty_midi.PrettyMIDI(file_path)

                        # data for fingertest
                        if "ft" in test.lower():
                            # Remove notes that start after 30 seconds
                            for instrument in midi.instruments:
                                instrument.notes = [note for note in instrument.notes if note.start <= 30.0]

                            # Extract played notes (non-drum, sorted by start time)
                            played_notes = self.extract_played_notes(midi=midi)

                            # prepare the data for DataFrame
                            info = {
                                'Participant_ID': participant_id,
                                'Test': test,
                                'Notes': played_notes,
                            }
                            data_fingertest.append(info)

                        else:
                            # Extract played notes (non-drum, sorted by start time)
                            played_notes = self.extract_played_notes(midi=midi)  
                                    
                            info = {
                                'Participant_ID': participant_id,
                                'Test': test,
                                'Notes': played_notes,
                            }
                            data_states.append(info)

                        print(f"✅ Loaded: {file_path}")
                    except Exception as e:
                        print(f"❌ Failed to load {file_path}: {e}")

        return (data_fingertest, data_states)



# df_fingertest = pd.DataFrame(data_fingertest)
# df_states = pd.DataFrame(data_states)

# df_fingertest.to_csv(output_dir / "fingertest.csv", index=False)
# df_states.to_csv(output_dir / "states.csv", index=False)

# print(f"Saved CSV files to: {output_dir}")