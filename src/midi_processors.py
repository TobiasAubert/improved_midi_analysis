from constants import (
    aufwärmen,
    block,
    pre_post_test,
    STATE_DEFS,
    TRANSITION_FREQUENCIES,
)
import pandas as pd
import numpy as np
from typing import Any
from numpy.lib.stride_tricks import sliding_window_view
from collections import Counter


class FingerdexProcessor:
    def process_fingerdata(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Process fingertest data.

        Args:
            data: iterable of entries where each entry is a dict with keys:
                  - 'participant_id'
                  - 'test'
                  - 'notes': list of note dicts with
                        'pitch'
                        'name':
                        'start':
                        'end':

        Returns:
            List of dicts with summary metrics per entry.
        """
        processed_data = []
        expected_sequence = ["F4", "C4", "E4", "D4", "F4"]
        sequence_len = len(expected_sequence)

        for entry in data:
            # Read the fields needed for the summary row.
            participant_id = entry.get("participant_id")
            test = entry.get("test")

            notes = entry.get("notes", []) or []
            # Notes are stored as dictionaries with at least a 'name' key.
            played_notes_name = [
                n.get("name") for n in notes if isinstance(n, dict) and "name" in n
            ]

            # Count how often the expected note pattern appears in order.
            correct_sequences = 0
            for i in range(len(played_notes_name) - sequence_len + 1):
                if played_notes_name[i : i + sequence_len] == expected_sequence:
                    correct_sequences += 1

            info = {
                "participant_id": participant_id,
                "test": test,
                "keystrokes": len(played_notes_name),
                "correct_sequences": correct_sequences,
                "notes": notes
            }
            processed_data.append(info)

        return pd.DataFrame(processed_data)
    
    def fix_wrong_seq(self, df: pd.DataFrame, seq_len: int = 5) -> pd.DataFrame:
        """Recompute the most common contiguous sequence of length ``seq_len``
        for rows where the recorded `correct_sequences` is zero and write the
        recalculated count back into the DataFrame.

        The function returns a copy of the input DataFrame with updated
        `correct_sequences` for affected rows and a new column
        `most_common_sequence` (list) holding the most frequent tuple (if any).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("fix_wrong_seq expects a pandas DataFrame")

        out = df.copy()

        notes_col = "notes"

        # Ensure notes column contains lists
        out[notes_col] = out[notes_col].apply(lambda x: x if isinstance(x, list) else [])

        def most_common_k_tuple_from_notes(notes: list[dict], k: int):
            names = [n.get("name") for n in notes if isinstance(n, dict) and "name" in n]
            n = len(names)
            if k <= 0 or k > n:
                return None, 0
            windows = (tuple(names[i : i + k]) for i in range(n - k + 1))
            c = Counter(windows)
            if not c:
                return None, 0
            seq, occurrences = c.most_common(1)[0]
            return seq, occurrences

        # Prepare a column for the detected most common sequence
        out["most_common_sequence"] = None

        mask = out["correct_sequences"] == 0        # nur die Zeilen, die neu berechnet werden sollen
        
        out["most_common_sequence"] = None
        
        for idx in out[mask].index:
            seq, count = most_common_k_tuple_from_notes(out.at[idx, notes_col], seq_len)
            if seq is None:
                out.at[idx, "correct_sequences"] = 0
                out.at[idx, "most_common_sequence"] = None
            else:
                out.at[idx, "correct_sequences"] = int(count)
                out.at[idx, "most_common_sequence"] = list(seq)
        return out
            
            

            

        
class StatesProcessor:
    """Detect state sequences and derive transition metrics from state timings."""

    def process_statedata(self, data):
        """Detect the expected state sequence for each entry.

        The output keeps the participant and test metadata and adds a
        ``detected_states`` list with note timing and index information.
        """
        processed_data = []
        for entry in data:
            # Choose the target sequence from the test label.
            test_type = str(entry.get("test")).lower()
            if "a" in test_type:
                target_sequence = aufwärmen
            elif test_type.startswith("b") or "block" in test_type:
                target_sequence = block
            elif "pre" in test_type or "post" in test_type:
                target_sequence = pre_post_test
            else:
                target_sequence = []

            detected_states = []
            current_note_idx = 0

            # Limit the search range so matching stays local.
            max_search_range = 20

            # Walk through the expected states in order.
            found_state_ranges = []
            for state_index, target_state_id in enumerate(target_sequence):
                required_pitches = STATE_DEFS[target_state_id]

                # Search ahead only from the current position.
                search_limit = min(
                    current_note_idx + max_search_range, len(entry.get("notes", []))
                )

                for i in range(current_note_idx, search_limit):
                    # Use a fixed note window to detect the required pitch set.
                    window = entry.get("notes", [])[i : i + 10]
                    window_pitches = {n["pitch"] for n in window}

                    if required_pitches.issubset(window_pitches):
                        # State found: keep only the notes that belong to the target pitches.
                        seen_pitches = set()
                        state_notes = []
                        for n in window:
                            if (
                                n["pitch"] in required_pitches
                                and n["pitch"] not in seen_pitches
                            ):
                                state_notes.append(n)
                                seen_pitches.add(n["pitch"])

                        # Sort by time so the start and end timestamps are stable.
                        state_notes = sorted(state_notes, key=lambda x: x["start"])

                        # Map the detected notes back to their positions in the original note list.
                        state_note_indices = [
                            entry.get("notes", []).index(n) for n in state_notes
                        ]
                        state_pitches = [n["pitch"] for n in state_notes]

                        detected_states.append(
                            {
                                "state": target_state_id,
                                "notes": state_notes,
                                "start_time": state_notes[0]["start"],
                                "end_time": state_notes[-1]["end"],
                                "target_index": state_index,  # Position in der Soll-Sequenz
                                "notes_skipped": i
                                - current_note_idx,  # WICHTIG: Wie viele Noten waren "Müll" dazwischen?
                                "state_info": [
                                    target_state_id,
                                    state_pitches,
                                    state_note_indices,
                                ],
                            }
                        )

                        # Move the cursor past the last note of this state so it is not reused.
                        first_note_in_state = min(state_note_indices)
                        last_note_in_state = max(state_note_indices)
                        found_state_ranges.append(
                            f"State {target_state_id}: played_notes Index {first_note_in_state} to {last_note_in_state} (Position {state_index})"
                        )
                        current_note_idx = last_note_in_state + 1
                        break

            # Assemble the row that will later become one DataFrame record.
            info = {
                "participant_id": entry["participant_id"],
                "test": entry["test"],  # example: Fingertest1, B2, Pretest
                "detected_states": self.remove_duplicate_states(detected_states),
                "notes": entry["notes"]
            }

            processed_data.append(info)

        return pd.DataFrame(processed_data)

    # Control: Remove duplicate states, keeping only the last occurrence of consecutive identical states
    # to do imbed this into the analyse state
    @staticmethod
    def remove_duplicate_states(
        states_list: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Keep only the last entry of consecutive repeated states."""
        if not states_list:
            return []

        filtered = []
        for i in range(len(states_list)):
            current_state = states_list[i]["state"]

            # Prüfen, ob dies das letzte Element ist ODER ob der nächste Zustand anders ist
            if (
                i == len(states_list) - 1
                or current_state != states_list[i + 1]["state"]
            ):
                filtered.append(states_list[i])

        return filtered

    def calculate_transitions(self, states_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute transitions for each row in a DataFrame and add a 'transitions' column.

        The function returns a new DataFrame with the same rows and one additional
        column `transitions` where each entry is a list of dicts with the keys:
        'position','from_state','to_state','transition_code','frequency',
        'onset_to_onset','offset_to_onset','state_duration','overlap'.
        """
        if not isinstance(states_df, pd.DataFrame):
            raise TypeError("calculate_transitions expects a pandas DataFrame")

        out_df = states_df.copy()

        # Build one transition list per row so the original DataFrame is unchanged.
        transitions_per_row = []
        for _, row in states_df.iterrows():
            states_list = row.get("detected_states") or []
            row_transitions = []
            if isinstance(states_list, list) and len(states_list) >= 2:
                # Ignore any state entries that are missing timing information.
                valid_states = [
                    s for s in states_list if "start_time" in s and "end_time" in s
                ]
                for i in range(len(valid_states) - 1):
                    s1 = valid_states[i]
                    s2 = valid_states[i + 1]

                    onset_to_onset = s2["start_time"] - s1["start_time"]
                    offset_to_onset = s2["start_time"] - s1["end_time"]
                    state_duration = s1["end_time"] - s1["start_time"]
                    overlap = max(0, s1["end_time"] - s2["start_time"])

                    transition_code = f"{s1['state']}{s2['state']}"
                    frequency = TRANSITION_FREQUENCIES.get(transition_code, "")

                    row_transitions.append(
                        {
                            "position": i,
                            "from_state": s1["state"],
                            "to_state": s2["state"],
                            "transition_code": transition_code,
                            "frequency": frequency,
                            "onset_to_onset": round(onset_to_onset, 4),
                            "offset_to_onset": round(offset_to_onset, 4),
                            "state_duration": round(state_duration, 4),
                            "overlap": round(overlap, 4),
                        }
                    )

            transitions_per_row.append(row_transitions)

        out_df["transitions"] = transitions_per_row
        return out_df

    def check_state_count(self, df: pd.DataFrame) -> None:
        df = df.copy()
        incorrect_rows = []
        for _, entry in df.iterrows():
            test = str(entry.get("test", "")).upper()
            ds = entry.get("detected_states") or []

            n = entry.get("notes") or []

            count_detected_states = len(ds)
            except_detected_states = 0
            if test == "A":
                except_detected_states = len(aufwärmen)
            elif test == "POST" or test == "PRE":
                except_detected_states = len(pre_post_test)
            elif "B" in test:
                except_detected_states = len(block)
            
            if except_detected_states - count_detected_states != 0:
                row = entry.to_dict()

                #list of all states in found in seq
                row["state_array"] = [
                    state_entry.get("state")
                    for state_entry in ds
                    if isinstance(state_entry, dict)
                ]

                #difference between expected and found states
                row["state_diff"] = except_detected_states - count_detected_states

                inf = [
                state_entry.get("state_info")
                for state_entry in ds
                if isinstance(state_entry, dict)
                ]

                # notes after last state
                if inf and len(inf[-1]) > 2:
                    state_note_indices_last = inf[-1][2]
                    biggest_index = max(state_note_indices_last)
                else:
                    state_note_indices_last = []
                    biggest_index = -1
                
                row["notes_after_last_state"] = [
                    note_entry.get("pitch")
                    for note_index, note_entry in enumerate(n)
                    if isinstance(note_entry, dict) and note_index > biggest_index
                ]

                # Collect all notes for every detected state as (state, notes) tuples.
                notes_all_state = []
                for state_info in inf:
                    if not isinstance(state_info, list) or len(state_info) < 3:
                        continue

                    state_note_indices = [
                        note_index
                        for note_index in state_info[2]
                        if isinstance(note_index, int)
                    ]
                    if not state_note_indices:
                        continue

                    first_note = min(state_note_indices)
                    last_note = max(state_note_indices)

                    state_notes = [
                        note_entry.get("pitch")
                        for note_index, note_entry in enumerate(n)
                        if (
                            isinstance(note_entry, dict)
                            and first_note <= note_index <= last_note
                        )
                    ]
                    notes_all_state.append((state_info[0], state_notes))

                row["notes_all_state"] = notes_all_state

                #all notes in seq
                notes = [
                    note_entry.get("pitch")
                    for note_entry in n
                    if isinstance(note_entry, dict)
                ]

                row["notes"] = notes

                #check if nots went missing because of iteration search window
                detected_state_notes = [
                    pitch
                    for _, state_note_list in notes_all_state
                    for pitch in state_note_list
                ]
                row["missing_note"] = detected_state_notes != notes

                incorrect_rows.append(row)

        if not incorrect_rows:
            print("all good")
            return

        filtered_df = pd.DataFrame(incorrect_rows)
        filtered_df.to_csv("state_count.csv", index=False)
        # filtered_df.to_csv("filtered.csv", index=False)
        filtered_df[["participant_id", "test", "state_diff", "state_array"]].to_csv("filtered.csv", index=False)
        filtered_df[["participant_id", "test", "state_diff", "notes"]].to_csv("notes.csv", index=False)
        filtered_df[["participant_id", "test", "notes_after_last_state"]].to_csv("notes_after_last_state.csv", index=False)
        filtered_df[["participant_id", "test", "missing_note", "notes_all_state"]].to_csv("notes_all_state.csv", index=False)


    def remove_incomplete_seq(self, df:pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            keep_rows = []

            for idx, entry in df.iterrows():
                test = str(entry.get("test", "")).upper()
                n = entry.get("notes") or []

                except_detected_states = 0
                if test == "A":
                    except_detected_states = len(aufwärmen)
                elif test == "POST" or test == "PRE":
                    except_detected_states = len(pre_post_test)
                elif "B" in test:
                    except_detected_states = len(block)

                notes= [
                        note_entry.get("pitch")
                        for note_entry in n
                        if isinstance(note_entry, dict)
                    ]
    
                if((len(notes)/ 4) >= except_detected_states):
                        keep_rows.append(idx)

            return df.loc[keep_rows].copy()


                
                                
                

