class FingerdexProcessor:
    def analyse_fingertest(self, data):
        """
        Process fingertest data.

        Args:
            data: iterable of entries where each entry is a dict with keys:
                  - 'Participant_ID' or 'Paticipant_ID'
                  - 'Test'
                  - 'Notes': list of note dicts with at least the 'name' key (e.g. 'F4')

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
    


