import os
import pretty_midi
import pandas as pd
from pathlib import Path


# Change this to your actual folder path or copy data in this folder
SCRIPT_DIR = Path(__file__).resolve().parent
root_folder = (SCRIPT_DIR / "../../data/midi_recordings").resolve()
output_dir = (SCRIPT_DIR / "../../data").resolve()
data_fingertest = []
data_states = []

def load_midi():
    # Walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(('.mid', '.midi')):
                file_path = os.path.join(dirpath, filename)

                try:
                    # Extract participant ID, test_name
                    participant_id = filename.split('_')[0]
                    test = filename.split('_')[1].split('.')[0]

                    # Extract note keystrokes
                    # Load MIDI file
                    midi = pretty_midi.PrettyMIDI(file_path)

                    # data for fingertest
                    if "ft" in test.lower():
                        # Remove notes that start after 30 seconds
                        for instrument in midi.instruments:
                            instrument.notes = [note for note in instrument.notes if note.start <= 30.0]

                        # Your expected pattern (adjust to match what they were supposed to play)
                        expected_sequence = ['F4', 'C4', 'E4', 'D4', 'F4']
                        sequence_len = len(expected_sequence)

                        # Extract played notes (non-drum, sorted by start time)
                        played_notes = []
                        for instrument in midi.instruments:
                            if not instrument.is_drum:
                                sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
                                for note in sorted_notes:
                                    name = pretty_midi.note_number_to_name(note.pitch)
                                    played_notes.append(name)


                        # Count correct sequences using a sliding window
                        correct_sequences = 0
                        for i in range(len(played_notes) - sequence_len + 1):
                            if played_notes[i:i+sequence_len] == expected_sequence:
                                correct_sequences += 1


                        # prepare the data for DataFrame
                        info = {
                            'Participant_ID': participant_id,
                            'Test': test,
                            'Keystrokes': len(played_notes),
                            'Correct_Sequences': correct_sequences,
                        }

                        data_fingertest.append(info)

                    else:
                        # - Store extracted pitches in an array named 'played_notes'
                        # Extract played notes (non-drum, sorted by start time)
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
                                
                        info = {
                            'Paticipant_ID': participant_id,
                            'Test': test,
                            'played_notes': played_notes,
                        }
                        
                        data_states.append(info)

                    print(f"✅ Loaded: {file_path}")
                except Exception as e:
                    print(f"❌ Failed to load {file_path}: {e}")

    return (data_fingertest, data_states)



df_fingertest = pd.DataFrame(data_fingertest)
df_states = pd.DataFrame(data_states)

df_fingertest.to_csv(output_dir / "fingertest.csv", index=False)
df_states.to_csv(output_dir / "states.csv", index=False)

print(f"Saved CSV files to: {output_dir}")