from constants import aufwärmen, block, pre_post_test, STATE_DEFS, TRANSITION_FREQUENCIES
import pandas as pd
import numpy as np

class FingerdexProcessor:
    def analyse_fingertest(self, data):
        """
        Process fingertest data.

        Args:
            data: iterable of entries where each entry is a dict with keys:
                  - 'Participant_ID' or 'Paticipant_ID'
                  - 'Test'
                  - 'Notes': list of note dicts with 
                        'pitch'
                        'name':
                        'start':
                        'end':

        Returns:
            List of dicts with summary metrics per entry.
        """
        processed_data = []
        expected_sequence = ['F4', 'C4', 'E4', 'D4', 'F4']
        sequence_len = len(expected_sequence)

        for entry in data:
            # robustly retrieve participant id and test
            participant_id = entry.get('Participant_ID')
            test = entry.get('Test')

            notes = entry.get('Notes', []) or []
            # notes is expected to be a list of dicts like {'name': 'F4', 'pitch': 65, ...}
            played_notes_name = [n.get('name') for n in notes if isinstance(n, dict) and 'name' in n]

            # Count correct sequences using a sliding window
            correct_sequences = 0
            for i in range(len(played_notes_name) - sequence_len + 1):
                if played_notes_name[i:i+sequence_len] == expected_sequence:
                    correct_sequences += 1

            info = {
                'Participant_ID': participant_id,
                'Test': test,
                'Keystrokes': len(played_notes_name),
                'Correct_Sequences': correct_sequences,
            }
            processed_data.append(info)

        return processed_data
    
class StatesProcessor:
    def analyse_states(self, data):
        processed_data = []
        for entry in data:     
            # 1. Bestimme die Ziel-Sequenz basierend auf dem Test-Namen
            test_type = str(entry.get('Test')).lower()
            if 'a' in test_type:
                target_sequence = aufwärmen
            elif test_type.startswith('b') or 'block' in test_type:
                target_sequence = block
            elif 'pre' in test_type or 'post' in test_type:
                target_sequence = pre_post_test
            else:
                target_sequence = []

            detected_states = []
            current_note_idx = 0
            
            max_search_range = 20  # Wie viele Noten darf das Skript "durchsuchen", um den State zu finden?
            
            # Wir gehen die Ziel-Zustände nacheinander durch
            mismatch_positions = []
            found_state_ranges = []
            for state_index, target_state_id in enumerate(target_sequence):
                required_pitches = STATE_DEFS[target_state_id]
                
                # Wir begrenzen die Suche auf einen Bereich ab der aktuellen Note
                search_limit = min(current_note_idx + max_search_range, len(entry.get('Notes', [])))
                
                for i in range(current_note_idx, search_limit - 5):
                    # Kleines Fenster von 10 Noten für die Erkennung eines 6er-Akkords
                    window = entry.get('Notes', [])[i : i + 10]
                    window_pitches = {n['pitch'] for n in window}
                    
                    if required_pitches.issubset(window_pitches):
                        # State gefunden! Extrahiere die relevanten Noten
                        seen_pitches = set()
                        state_notes = []
                        for n in window:
                            if n['pitch'] in required_pitches and n['pitch'] not in seen_pitches:
                                state_notes.append(n)
                                seen_pitches.add(n['pitch'])
                        
                        # Sortiere state_notes nach Zeit, um korrekte Start/End-Zeiten zu haben
                        state_notes = sorted(state_notes, key=lambda x: x['start'])
                        
                        # Get indexes of these notes in played_notes
                        state_note_indices = [entry.get('Notes', []).index(n) for n in state_notes]
                        state_pitches = [n['pitch'] for n in state_notes]
                        
                        detected_states.append({
                            'state': target_state_id,
                            'notes': state_notes,
                            'start_time': state_notes[0]['start'],
                            'end_time': state_notes[-1]['end'],
                            'target_index': state_index, # Position in der Soll-Sequenz
                            'notes_skipped': i - current_note_idx,  # WICHTIG: Wie viele Noten waren "Müll" dazwischen?
                            'state_info': [target_state_id, state_pitches, state_note_indices]  # [state, pitches, indexes]
                        })
                        
                        # Setze den Zeiger hinter die letzte Note dieses erkannten States
                        # damit wir nicht dieselben Noten für den nächsten State nutzen
                        first_note_in_state = min(state_note_indices)
                        last_note_in_state = max(state_note_indices)
                        found_state_ranges.append(
                            f"State {target_state_id}: played_notes Index {first_note_in_state} to {last_note_in_state} (Position {state_index})"
                        )
                        current_note_idx = last_note_in_state + 1
                        break

            if mismatch_positions:
                last_found_pitch = None
                last_found_idx = None
                mismatch_output_lines = []
                for state_entry in reversed(detected_states):
                    if state_entry.get('state') is not None and state_entry.get('notes'):
                        last_note = state_entry['notes'][-1]
                        last_found_pitch = last_note.get('pitch')
                        last_found_idx = entry.get('Notes', []).index(last_note)
                        break

                for target_state_id, state_index in mismatch_positions:
                    if last_found_pitch is not None:
                        line = (
                            f"   ℹ️ State {target_state_id} an Position {state_index} nicht gefunden. "
                            f"Letzter Pitch: {last_found_pitch} (played_notes Index {last_found_idx})"
                        )
                    else:
                        line = (
                            f"   ℹ️ State {target_state_id} an Position {state_index} nicht gefunden. "
                            "Letzter Pitch: n/a"
                        )
                    print(line)
                    mismatch_output_lines.append(line)

                if found_state_ranges:
                    header = "   🔎 Gefundene States (Index-Bereiche):"
                    # print(header)
                    mismatch_output_lines.append(header)
                    for entry in found_state_ranges:
                        line = f"   - {entry}"
                        # print(line)
                        mismatch_output_lines.append(line)

            # Calculate last found pitch and index for DataFrame
            last_found_pitch = None
            last_found_idx = None
            for state_entry in reversed(detected_states):
                if state_entry.get('state') is not None and state_entry.get('notes'):
                    last_note = state_entry['notes'][-1]
                    last_found_pitch = last_note.get('pitch')
                    last_found_idx = entry.get('Notes', []).index(last_note)
                    break

            # prepare the data for DataFrame
            info = {
                'Participant_ID': entry['Participant_ID'],
                'Test': entry['Test'], #example: Fingertest1, B2, Pretest
                'detected_states': detected_states,
                # 'last_found_pitch': last_found_pitch,
                # 'last_found_idx': last_found_idx,
                # 'all_pitches': [n['pitch'] for n in played_notes],  # Liste von Werten
                # 'Keystrokes': len(played_notes),
                # 'Correct_Sequences': correct_sequences,
            }

            processed_data.append(info)

        return processed_data
    
    #Control: Remove duplicate states, keeping only the last occurrence of consecutive identical states
    # to do imbed this into the analyse state 
    @staticmethod
    def remove_duplicate_states(states_list):
        if not states_list:
            return []
        
        filtered = []
        for i in range(len(states_list)):
            current_state = states_list[i]['state']
            
            # Prüfen, ob dies das letzte Element ist ODER ob der nächste Zustand anders ist
            if i == len(states_list) - 1 or current_state != states_list[i + 1]['state']:
                filtered.append(states_list[i])
                
        return filtered
    


    def calculate_transitions(self, states_df):
        """
        Compute transitions for each row in a DataFrame and add a 'transitions' column.

        The function returns a new DataFrame with the same rows and one additional
        column `transitions` where each entry is a list of dicts with the keys:
        'position','from_state','to_state','transition_code','frequency',
        'onset_to_onset','offset_to_onset','state_duration','overlap'.
        """
        if not isinstance(states_df, pd.DataFrame):
            raise TypeError("calculate_transitions expects a pandas DataFrame")

        transitions_per_row = []
        for _, row in states_df.iterrows():
            states_list = row.get('detected_states') or []
            row_transitions = []
            if isinstance(states_list, list) and len(states_list) >= 2:
                valid_states = [s for s in states_list if 'start_time' in s and 'end_time' in s]
                for i in range(len(valid_states) - 1):
                    s1 = valid_states[i]
                    s2 = valid_states[i + 1]

                    onset_to_onset = s2['start_time'] - s1['start_time']
                    offset_to_onset = s2['start_time'] - s1['end_time']
                    state_duration = s1['end_time'] - s1['start_time']
                    overlap = max(0, s1['end_time'] - s2['start_time'])

                    transition_code = f"{s1['state']}{s2['state']}"
                    frequency = TRANSITION_FREQUENCIES.get(transition_code, "")

                    row_transitions.append({
                        'position': i,
                        'from_state': s1['state'],
                        'to_state': s2['state'],
                        'transition_code': transition_code,
                        'frequency': frequency,
                        'onset_to_onset': round(onset_to_onset, 4),
                        'offset_to_onset': round(offset_to_onset, 4),
                        'state_duration': round(state_duration, 4),
                        'overlap': round(overlap, 4),
                    })

            transitions_per_row.append(row_transitions)

        out_df = states_df.copy()
        out_df['transitions'] = transitions_per_row
        return out_df