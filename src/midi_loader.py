import os
import pretty_midi
import pandas as pd
from pathlib import Path
from typing import Any


class MIDILoader:
    def __init__(self):
        pass

    def extract_midi_info(self, filename: str) -> tuple[str, str]:
        """Extract the participant ID and test name from a MIDI filename.

        Args:
            filename: MIDI filename in the form ``PARTICIPANT_TEST.mid``.

        Returns:
            A tuple containing ``participant_id`` and ``test``.
        """
        participant_id = filename.split("_")[0]
        test = filename.split("_")[1].split(".")[0]  # A, B1, POST, PRE / FT1
        return (participant_id, test)

    def extract_played_notes(
        self, midi: pretty_midi.PrettyMIDI, exclude_drums: bool = True
    ) -> list[dict[str, Any]]:
        """Extract played notes from a PrettyMIDI object.

        Args:
            midi: Parsed MIDI file.
            exclude_drums: Kept for compatibility; drum tracks are ignored.

        Returns:
            A list of note dictionaries with pitch, name, start, and end.
        """
        played_notes: list[dict[str, Any]] = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
                for note in sorted_notes:
                    name = pretty_midi.note_number_to_name(note.pitch)
                    played_notes.append(
                        {
                            "pitch": note.pitch,
                            "name": name,
                            "start": note.start,
                            "end": note.end,
                        }
                    )
        return played_notes

    def load_midi(
        self, root_folder: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Load MIDI files from a folder and split them into two datasets.

        Args:
            root_folder: Directory that contains participant subfolders with MIDI files.

        Returns:
            A tuple of ``(data_fingertest, data_states)``.
        """
        data_fingertest = []
        data_states = []
        # Walk through all subfolders
        for dirpath, dirnames, filenames in os.walk(root_folder):
            for filename in filenames:
                if filename.lower().endswith((".mid", ".midi")):
                    file_path = os.path.join(dirpath, filename)

                    try:
                        participant_id, test = self.extract_midi_info(filename=filename)

                        # Extract note keystrokes
                        midi = pretty_midi.PrettyMIDI(file_path)

                        # data for fingertest
                        if "ft" in test.lower():
                            # Remove notes that start after 30 seconds
                            for instrument in midi.instruments:
                                instrument.notes = [
                                    note
                                    for note in instrument.notes
                                    if note.start <= 30.0
                                ]

                            # Extract played notes (non-drum, sorted by start time)
                            played_notes = self.extract_played_notes(midi=midi)

                            # prepare the data for DataFrame
                            info = {
                                "participant_id": participant_id,
                                "test": test,
                                "notes": played_notes,
                            }
                            data_fingertest.append(info)

                        else:
                            # Extract played notes (non-drum, sorted by start time)
                            played_notes = self.extract_played_notes(midi=midi)

                            info = {
                                "participant_id": participant_id,
                                "test": test,
                                "notes": played_notes,
                            }
                            data_states.append(info)

                        print(f"✅ Loaded: {file_path}")
                    except Exception as e:
                        print(f"❌ Failed to load {file_path}: {e}")

        return (data_fingertest, data_states)
