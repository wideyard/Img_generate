import random

def generate_pitches_for_key(key_note, key_mode):
    key_signatures = {
        ('c', '\\major'): {}, ('g', '\\major'): {'f': 'fis'}, ('d', '\\major'): {'f': 'fis', 'c': 'cis'},
        ('a', '\\major'): {'f': 'fis', 'c': 'cis', 'g': 'gis'},
        ('f', '\\major'): {'b': 'bes'}, ('bes', '\\major'): {'b': 'bes', 'e': 'ees'},
        ('ees', '\\major'): {'b': 'bes', 'e': 'ees', 'a': 'aes'},
        ('e', '\\major'): {'f': 'fis', 'c': 'cis', 'g': 'gis', 'd': 'dis'},
        ('a', '\\minor'): {}, ('e', '\\minor'): {'f': 'fis'}, ('b', '\\minor'): {'f': 'fis', 'c': 'cis'},
        ('fis', '\\minor'): {'f': 'fis', 'c': 'cis', 'g': 'gis'},
        ('d', '\\minor'): {'b': 'bes'}, ('g', '\\minor'): {'b': 'bes', 'e': 'ees'},
        ('c', '\\minor'): {'b': 'bes', 'e': 'ees', 'a': 'aes'},
        ('cis', '\\minor'): {'f': 'fis', 'c': 'cis', 'g': 'gis', 'd': 'dis'},
    }

    accidentals = key_signatures.get((key_note, key_mode), {})

    base_notes = ['c', 'd', 'e', 'f', 'g', 'a', 'b']
    all_pitches = []

    for note in ['e', 'f', 'g', 'a', 'b']:
        all_pitches.append(accidentals.get(note, note) + ",")

    for octave_mark in ["", "'"]:
        for note in base_notes:
            all_pitches.append(accidentals.get(note, note) + octave_mark)

    for note in ['c', 'd', 'e', 'f', 'g', 'a', 'b']:
        all_pitches.append(accidentals.get(note, note) + "''")

    diatonic_map = {n: n for n in base_notes}
    for natural_note, accidental_note in accidentals.items():
        diatonic_map[natural_note] = accidental_note

    all_pitches = []
    octave_marks = [",", "", "'", "''"]
    base_notes_in_order = ["c", "d", "e", "f", "g", "a", "b"]

    for mark in octave_marks:
        for note in base_notes_in_order:
            pitch = diatonic_map.get(note, note) + mark
            all_pitches.append(pitch)

    start_index = all_pitches.index(diatonic_map['e'] + ",")
    end_index = all_pitches.index(diatonic_map['a'] + "''")

    return all_pitches[start_index : end_index + 1]


def generate_random_score():
    clef = random.choice(["treble", "bass"])

    time_signatures = ["2/2", "3/4", "4/4", "6/8", "2/4"]
    time_signature = random.choice(time_signatures)

    keys = [
        ('c', '\\major'), ('a', '\\minor'),
        ('g', '\\major'), ('e', '\\minor'),
        ('d', '\\major'), ('b', '\\minor'),
        ('a', '\\major'), ('fis', '\\minor'),
        ('f', '\\major'), ('d', '\\minor'),
        ('bes', '\\major'), ('g', '\\minor'),
        ('ees', '\\major'), ('c', '\\minor'),
        ('e', '\\major'), ('cis', '\\minor')
    ]
    key_note, key_mode = random.choice(keys)

    tempo_data = {
        "Grave": (20, 40),
        "Largo": (40, 60),
        "Larghetto": (60, 66),
        "Adagio": (60, 76),
        "Andante": (76, 108),
        "Andantino": (70, 90),
        "Moderato": (108, 120),
        "Allegretto": (100, 120),
        "Allegro": (120, 168),
        "Vivace": (140, 176),
        "Presto": (168, 200),
        "Prestissimo": (200, 240),
        "None": None,
    }

    selected_description = random.choice(list(tempo_data.keys()))
    bpm_range = tempo_data[selected_description]

    if bpm_range is None:
        tempo_mark = ""
    else:
        random_bpm = random.randint(*bpm_range)
        tempo_mark = f'\\tempo "{selected_description}" 4 = {random_bpm}'

    num_measures = random.randint(3, 5)

    notes_string = ""
    numerator, denominator = map(int, time_signature.split('/'))
    measure_duration = numerator / denominator

    base_durations = [16, 8, 4, 2, 1]

    all_notes = []
    for i in range(num_measures):
        current_measure_duration = 0
        measure_notes = []
        while current_measure_duration < measure_duration:
            remaining_duration = measure_duration - current_measure_duration

            possible_durations = [d for d in base_durations if 1/d <= remaining_duration]
            if not possible_durations:
                break

            duration_val = random.choice(possible_durations)
            note_duration = 1 / duration_val

            is_dotted = random.random() < 0.1
            if is_dotted and (note_duration * 1.5) <= remaining_duration:
                duration_str = f"{duration_val}."
                note_duration *= 1.5
            else:
                duration_str = str(duration_val)

            all_pitches = generate_pitches_for_key(key_note, key_mode)

            if clef == "treble":
                pitch_range = all_pitches[12:]
            else:
                pitch_range = all_pitches[:13]

            def add_accidental(pitch):
                if random.random() < 0.1:
                    if "is" in pitch or "es" in pitch:
                        return pitch
                    if random.random() < 0.5:
                        return pitch[0] + "is" + pitch[1:]
                    else:
                        return pitch[0] + "es" + pitch[1:]
                return pitch

            r = random.random()
            if r < 0.1:
                num_notes_in_chord = 3
            elif r < 0.2:
                num_notes_in_chord = 2
            else:
                num_notes_in_chord = 1

            if num_notes_in_chord == 1:
                pitch = random.choice(pitch_range)
                pitch = add_accidental(pitch)
                if random.random() < 0.1:
                    note_or_chord = f"r{duration_str}"
                else:
                    note_or_chord = f"{pitch}{duration_str}"
            elif num_notes_in_chord == 2:
                note1_index_in_range = random.randrange(len(pitch_range))

                min_index2 = max(0, note1_index_in_range - 7)
                max_index2 = min(len(pitch_range) - 1, note1_index_in_range + 7)

                possible_indices2 = list(range(min_index2, max_index2 + 1))
                possible_indices2.remove(note1_index_in_range)

                note2_index_in_range = random.choice(possible_indices2)

                chord_pitches = [pitch_range[note1_index_in_range], pitch_range[note2_index_in_range]]
                chord_pitches = [add_accidental(p) for p in chord_pitches]

                chord_pitches_str = " ".join(chord_pitches)
                if random.random() < 0.05:
                    note_or_chord = f"r{duration_str}"
                else:
                    note_or_chord = f"<{chord_pitches_str}>{duration_str}"
            else:
                start_note_index = random.randrange(len(pitch_range) - 7)
                interval_indices = list(range(1, 8))
                note2_offset, note3_offset = random.sample(interval_indices, 2)

                note1_index_in_range = start_note_index
                note2_index_in_range = start_note_index + note2_offset
                note3_index_in_range = start_note_index + note3_offset

                chord_pitches = [pitch_range[note1_index_in_range], pitch_range[note2_index_in_range], pitch_range[note3_index_in_range]]
                chord_pitches = [add_accidental(p) for p in chord_pitches]

                chord_pitches_str = " ".join(chord_pitches)

                if random.random() < 0.1:
                    note_or_chord = f"r{duration_str}"
                else:
                    note_or_chord = f"<{chord_pitches_str}>{duration_str}"

            measure_notes.append(note_or_chord)
            current_measure_duration += note_duration
        all_notes.extend(measure_notes)

    slurred_notes = [False] * len(all_notes)
    num_slurs_to_add = random.randint(0, num_measures * 2)

    for _ in range(num_slurs_to_add):
        if len(all_notes) < 2:
            break

        for _ in range(10): # Try 10 times to find a valid slur position
            slur_len = random.randint(2, min(8, len(all_notes)))
            start_idx = random.randint(0, len(all_notes) - slur_len)
            end_idx = start_idx + slur_len - 1

            # Check for rests, bar lines, or existing slurs in the range
            is_valid_slur = True
            for i in range(start_idx, end_idx + 1):
                note = all_notes[i]
                if 'r' in note or slurred_notes[i]:
                    is_valid_slur = False
                    break

            if is_valid_slur:
                all_notes[start_idx] += "("
                all_notes[end_idx] += ")"
                for i in range(start_idx, end_idx + 1):
                    slurred_notes[i] = True
                break

    notes_string = " ".join(all_notes)

    score = f'\\header {{ tagline = "" }}\n' \
            f'\\version "2.20.0" {{ \\clef {clef} \\time {time_signature} \\key {key_note} {key_mode} ' \
            f'{tempo_mark}\n' \
            f'{notes_string.strip()} }}\n'

    return score

if __name__ == "__main__":
    random_score = generate_random_score()
    with open('./scores.txt', 'w') as out_file:
        out_file.write(random_score)
    lines = random_score.split('\n')[1:]
    print('\n'.join(lines))
