import random
from dataclasses import dataclass

import music21

# TODO list:
# 1. Revise evaluation functions
#    - more complex assessment of rhythmic variety of descant
#    - improve speed of counterpoint evaluation
# 2. Allow different key and time signatures
# 3. Allow input of XML file with melody and chord voicings
#    - congruence of descant with each voice


@dataclass(frozen=True)
class MelodyData:
    """
    A data class representing the data of a melody.

    This class encapsulates the details of a melody including its notes, total
    duration, and the number of bars. The notes are represented as a list of
    tuples, with each tuple containing a pitch and its duration. The total
    duration and the number of bars are computed based on the notes provided.

    Attributes:
        notes (list of tuples): List of tuples representing the melody's notes.
            Each tuple is in the format (pitch, duration).
        duration (int): Total duration of the melody, computed from notes.
        number_of_bars (int): Total number of bars in the melody, computed from
            the duration assuming a 4/4 time signature.

    Methods:
        __post_init__: A method called after the data class initialization to
            calculate and set the duration and number of bars based on the
            provided notes.
    """

    notes: list
    duration: int = None  # Computed attribute
    number_of_bars: int = None  # Computed attribute

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.notes)
        )
        object.__setattr__(self, "number_of_bars", self.duration // 4)


@dataclass(frozen=True)
class ChordData:
    """
    A data class representing the data of an accompaniment chord sequence.

    This class encapsulates the details of an accompaniment including its chords, 
    total duration, and the number of bars. The chords are represented as a list of
    tuples, with each tuple containing a chord symbol and its duration. The total
    duration and the number of bars are computed based on the chord sequence provided.

    Attributes:
        chords (list of tuples): List of tuples representing the chord sequence.
            Each tuple is in the format (chord symbol, duration).
        duration (int): Total duration of the accompaniment, computed from chords.
        number_of_bars (int): Total number of bars in the accompaniment, computed from
            the duration assuming a 4/4 time signature.

    Methods:
        __post_init__: A method called after the data class initialization to
            calculate and set the duration and number of bars based on the
            provided chords.
    """

    chords: list
    duration: int = None  # Computed attribute
    number_of_bars: int = None  # Computed attribute

    def __post_init__(self):
        object.__setattr__(
            self, "duration", sum(duration for _, duration in self.chords)
        )
        object.__setattr__(self, "number_of_bars", self.duration // 4)


class GeneticDescantGenerator:
    """
    Generates a descant/melody for a given chord sequence using a genetic algorithm.
    It evolves a population of melodies to find one that best fits the
    chord sequence based on a fitness function.

    Attributes:
        chord_data (ChordData): Data containing accompaniment chords.
        notes (list): Available notes for generating descant.
        population_size (int): Size of the descant population.
        mutation_rate (float): Probability of mutation in the genetic algorithm.
        fitness_evaluator (FitnessEvaluator): Instance used to assess fitness.
    """

    def __init__(
        self,
        chord_data,
        notes,
        population_size,
        mutation_rate,
        fitness_evaluator,
    ):
        """
        Initializes the generator with chord data, notes, population size,
        mutation rate, and a fitness evaluator.

        Parameters:
            chord_data (ChordData): Accompaniment information.
            notes (list): Available notes.
            population_size (int): Size of population in the algorithm.
            mutation_rate (float): Mutation probability per chord.
            fitness_evaluator (FitnessEvaluator): Evaluator for chord fitness.
        """
        self.chord_data = chord_data
        self.notes = notes
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.fitness_evaluator = fitness_evaluator
        self._population = []

    def generate(self, generations=1000):
        """
        Generates a descant for a given chord sequence using a genetic
        algorithm.

        Parameters:
            generations (int): Number of generations for evolution.

        Returns:
            best_descant (list): Descant with the highest fitness
                found in the last generation.
        """
        self._population = self._initialise_population()
        for x in range(generations):
            if (x + 1) % 100 == 0:
                print(f"Generation {x + 1} out of {generations}")
            parents = self._select_parents()
            new_population = self._create_new_population(parents)
            self._population = new_population
        best_descant = (
            self.fitness_evaluator.get_descant_with_highest_fitness(
                self._population
            )
        )
        return best_descant

    def _initialise_population(self):
        """
        Initializes population with random notes.

        Returns:
            list: List of randomly generated note sequences.
        """
        return [
            self._generate_random_notes()
            for _ in range(self.population_size)
        ]

    def _generate_random_notes(self):
        """
        Generate a random note sequence with the same duration
        as the accompaniment.

        Returns:
            list: List of randomly generated notes.
        """
        return [
            random.choice(self.notes)
            for _ in range(self.chord_data.duration)
        ]

    def _select_parents(self):
        """
        Selects parent sequences for breeding based on fitness.

        Returns:
            list: Selected parent note sequences.
        """
        fitness_values = [
            self.fitness_evaluator.evaluate(seq) for seq in self._population
        ]
        return random.choices(
            self._population, weights=fitness_values, k=self.population_size
        )

    def _create_new_population(self, parents):
        """
        Generates a new population of note sequences from the provided parents.

        This method creates a new generation of note sequences using crossover
        and mutation operations. For each pair of parent note sequences,
        it generates two children. Each child is the result of a crossover
        operation between the pair of parents, followed by a potential
        mutation. The new population is formed by collecting all these
        children.

        The method ensures that the new population size is equal to the
        predefined population size of the generator. It processes parents in
        pairs, and for each pair, two children are generated.

        Parameters:
            parents (list): A list of parent note sequences from which to
                generate the new population.

        Returns:
            list: A new population of note sequences, generated from the
                parents.

        Note:
            This method assumes an even population size and that the number of
            parents is equal to the predefined population size.
        """
        new_population = []
        for i in range(0, self.population_size, 2):
            child1, child2 = self._crossover(
                parents[i], parents[i + 1]
            ), self._crossover(parents[i + 1], parents[i])
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def _crossover(self, parent1, parent2):
        """
        Combines two parent sequences into a new child sequence using one-point
        crossover.

        Parameters:
            parent1 (list): First parent note sequence.
            parent2 (list): Second parent note sequence.

        Returns:
            list: Resulting child note sequence.
        """
        total_duration = 0 # Track total duration of child sequence
        child = [] # Store child sequence

        # Continue until the total duration is met
        while total_duration < self.chord_data.duration:
            # Randomly select a cut index for crossover,
            # ensuring it doesn't exceed either parent's length
            cut_index = random.randint(1, min(len(parent1), len(parent2)) - 1)
            segment = parent1[:cut_index] + parent2[cut_index:]
            segment_duration = sum(duration for _, duration in segment)

            if total_duration + segment_duration > self.chord_data.duration:
                # Adjust the last note's duration if it exceeds the total duration
                for note, duration in segment:
                    if total_duration + duration > self.chord_data.duration:
                        duration = self.chord_data.duration - total_duration
                    child.append((note, duration))
                    total_duration += duration
                    if total_duration >= self.chord_data.duration:
                        break
            else:
                child.extend(segment)
                total_duration += segment_duration

        return child

    def _mutate(self, note_sequence):
        """
        Mutates a note in the sequence based on mutation rate.

        Parameters:
            note_sequence (list): Note sequence to mutate.

        Returns:
            list: Mutated note sequence.
        """
        if random.random() < self.mutation_rate:
            mutation_index = random.randint(0, len(note_sequence) - 1)
            note_sequence[mutation_index] = random.choice(self.notes)
        return note_sequence


class FitnessEvaluator:
    """
    Evaluates the fitness of a note sequence based on various musical criteria.

    Attributes:
        chords (list): List of tuples representing chords as (chord symbol, duration).
        melody (list): List of tuples representing melody notes as (note name, duration).
        chord_mappings (dict): Dictionary of chords with their corresponding notes.
        notes (list): List of available notes for the descant.
        weights (dict): Weights for different fitness evaluation functions.
        preferred_transitions (dict): Preferred transitions between notes.
    """

    def __init__(
        self, chord_data, melody_data, chord_mappings, notes, weights, preferred_transitions
    ):
        """
        Initialize the FitnessEvaluator with accompaniment, melody, chord mappings,
        notes, weights, and preferred transitions.

        Parameters:
            chord_data (ChordData): Accompaniment information.
            melody_data (MelodyData): Melody information.
            chord_mappings (dict): Available chords mapped to their notes.
            notes (list): Available notes for the descant.
            weights (dict): Weights for each fitness evaluation function.
            preferred_transitions (dict): Preferred transitions between notes.
        """
        self.chord_data = chord_data
        self.melody_data = melody_data
        self.chord_mappings = chord_mappings
        self.notes = notes
        self.weights = weights
        self.preferred_transitions = preferred_transitions

    def get_descant_with_highest_fitness(self, note_sequences):
        """
        Returns the note sequence with the highest fitness score.

        Parameters:
            note_sequences (list): List of note sequences to evaluate.

        Returns:
            list: Note sequence with the highest fitness score.
        """
        return max(note_sequences, key=self.evaluate)

    def evaluate(self, note_sequence):
        """
        Evaluate the fitness of a given note sequence.

        Parameters:
            note_sequence (list): The note sequence to evaluate.

        Returns:
            float: The overall fitness score of the note sequence.
        """
        return sum(
            self.weights[func] * getattr(self, f"_{func}")(note_sequence)
            for func in self.weights
        )

    def _chord_descant_congruence(self, note_sequence):
        """
        Calculates the congruence between the chord sequence and the descant.
        This function assesses how well each chord in the sequence aligns
        with the corresponding segment of the melody. The alignment is
        measured by checking if the notes in the descant are present in the
        chords being played at the same time, rewarding sequences where the
        descant notes fit well with the chords.

        Parameters:
            note_sequence (list): A list of notes to be evaluated against the
                accompaniment.

        Returns:
            float: A score representing the degree of congruence between the
                chord sequence and the descant, normalized by the descant's
                duration.
        """
        score, descant_index = 0, 0
        for chord in self.chord_data.chords:
            bar_duration = 0
            while bar_duration < 4 and descant_index < len(note_sequence):
                pitch, duration = note_sequence[descant_index]
                if pitch[0] in self.chord_mappings[chord[0]]:
                    score += duration
                bar_duration += duration
                descant_index += 1
        return score / self.chord_data.duration
    
    def _counterpoint(self, note_sequence):
        """
        Measures the quality of counterpoint between the descant and the melody.
        Rewards consonances and penalizes dissonances.
        Rewards resolution of dissonances on the following beat.

        Parameters:
            note_sequence (list): A list of notes to be evaluated against the
                melody.

        Returns:
            float: A score representing the degree of congruence between the
                melody and the descant, normalized by the duration.
        """
        score = 0
        melody_index, descant_index = 0, 0
        melody_time, descant_time = 0, 0
        previous_dissonant = False

        # Progress through the melody and descant by quarter note beats
        while (melody_index < len(self.melody_data.notes)
               and descant_index < len(note_sequence)):
            melody_note, melody_duration = self.melody_data.notes[melody_index]
            descant_note, descant_duration = note_sequence[descant_index]

            # Find the interval between the current melody and descant notes
            melody_pitch = music21.pitch.Pitch(melody_note)
            descant_pitch = music21.pitch.Pitch(descant_note)
            interval = music21.interval.Interval(melody_pitch, descant_pitch)

            # Check if the interval is consonant or dissonant
            if interval.isConsonant():
                score += 1 # Consonant intervals are preferred
                if previous_dissonant:
                    score += 2 # Reward resolution of dissonance
            else:
                previous_dissonant = True
                score -= 1 # Penalize dissonant intervals
            
            # Update the indices and durations
            # If the descant note ends before the melody note,
            # move to the next descant note
            if descant_time + descant_duration < melody_time + melody_duration:
                descant_time += descant_duration
                descant_index += 1
            # If the notes end together, advance both indices
            elif descant_time + descant_duration == melody_time + melody_duration:
                descant_time += descant_duration
                descant_index += 1
                melody_time += melody_duration
                melody_index += 1
            # If the melody note ends before the descant note,
            # move to the next melody note
            else:
                melody_time += melody_duration
                melody_index += 1
        
        return score / self.melody_data.duration # Normalize by total duration

    def _note_variety(self, note_sequence):
        """
        Evaluates the diversity of notes used in the sequence. This function
        calculates a score based on the number of unique notes present in the
        sequence compared to the total available notes. Higher variety in the
        note sequence results in a higher score, promoting musical
        complexity and interest.

        Parameters:
            note_sequence (list): The note sequence to evaluate.

        Returns:
            float: A normalized score representing the variety of notes in the
                sequence relative to the total number of available notes.
        """
        unique_notes = len(set(note_sequence))
        total_notes = len(self.notes)
        return unique_notes / total_notes
    
    def _rhythmic_variety(self, note_sequence):
        """
        Evaluates the diversity of rhythms used in the sequence. This function
        calculates a score based on the number of unique rhythms present in the
        sequence. Higher variety in the note sequence results in a higher score,
        promoting musical complexity and interest.

        Parameters:
            note_sequence (list): The note sequence to evaluate.

        Returns:
            float: A normalized score representing the variety of rhythms in the
                sequence.
        """
        unique_rhythms = len(set(duration for _, duration in note_sequence))
        total_rhythms = len(set(duration for _, duration in self.notes))
        return unique_rhythms / total_rhythms

    def _voice_leading(self, note_sequence):
        """
        Assesses the voice leading of the note sequence by examining the
        transitions between successive notes. This function scores the
        sequence based on how frequently the note transitions align with
        predefined preferred transitions. Smooth and musically pleasant
        transitions result in a higher score.

        Parameters:
            note_sequence (list): The note sequence to evaluate.

        Returns:
            float: A normalized score based on the frequency of preferred note
                transitions in the sequence.
        """
        score = 0
        for i in range(len(note_sequence) - 1):
            next_note = note_sequence[i + 1][0]
            if next_note in self.preferred_transitions[note_sequence[i][0]]:
                score += 1
        return score / (len(note_sequence) - 1)

    def _functional_harmony(self, note_sequence):
        """
        Evaluates the note sequence based on principles of functional harmony.
        This function checks for the presence of key harmonic functions such as
        notes from the tonic chord at the beginning and end of the sequence.
        Adherence to these harmonic conventions is rewarded with a higher score.

        Parameters:
            note_sequence (list): The note sequence to evaluate.

        Returns:
            float: A score representing the extent to which the sequence
                adheres to traditional functional harmony, normalized by
                the number of checks performed.
        """
        score = 0
        if note_sequence[0][0][0] in ["C", "E", "G"]:
            score += 1
        if note_sequence[-1][0][0] in ["C", "E", "G"]:
            score += 1
        return score / 2


def create_score(descant, melody, chord_sequence, chord_mappings, instrument="violin"):
    """
    Create a music21 score with a given descant, melody, and chord sequence.

    Args:
        descant (list): A list of tuples representing notes in the format
            (note_name, duration).
        melody (list): A list of tuples representing notes in the format
            (note_name, duration).
        chord_sequence (list): A list of tuples representing chords
            in the format (chord_symbol, duration).
        chord_mappings (dict): Available chords mapped to their notes.
        instrument (str): The instrument to use for the descant.

    Returns:
        music21.stream.Score: A music score containing the descant and chord
            sequence.
    """
    # Create a Score object
    score = music21.stream.Score()

    # Create the descant part and add notes to it
    descant_part = music21.stream.Part()
    descant_part.append(music21.instrument.fromString(instrument))
    clef = music21.clef.TrebleClef() if instrument == "violin" else music21.clef.AltoClef()
    descant_part.append(clef)
    for note_name, duration in descant:
        descant_note = music21.note.Note(note_name, quarterLength=duration)
        descant_part.append(descant_note)

    # Create the melody part and add notes to it
    melody_part = music21.stream.Part()
    melody_part.append(music21.instrument.Vocalist())
    for note_name, duration in melody:
        melody_note = music21.note.Note(note_name, quarterLength=duration)
        melody_part.append(melody_note)

    # Create the chord part and add chords to it
    chord_part = music21.stream.Part()
    current_duration = 0  # Track the duration for chord placement

    for chord_name, duration in chord_sequence:
        # Translate chord names to note lists
        chord_notes_list = chord_mappings.get(chord_name, [])
        # Create a music21 chord
        chord_notes = music21.chord.Chord(
            chord_notes_list, quarterLength=duration
        )  # Assuming 4/4 time signature
        chord_notes.offset = current_duration
        chord_part.append(chord_notes)
        current_duration += duration  # Increase by 4 beats

    # Append parts to the score
    score.append(descant_part)
    score.append(melody_part)
    score.append(chord_part)

    return score


def main():
    jesus_loves_me_chords = [
        ("C", 4),
        ("C", 4),
        ("F", 4),
        ("C", 4),
        ("C", 4),
        ("C", 4),
        ("F", 2),
        ("C", 2),
        ("G", 2),
        ("C", 2),
        ("C", 4),
        ("F", 4),
        ("C", 4),
        ("G", 4),
        ("C", 4),
        ("F", 4),
        ("C", 2),
        ("G", 2),
        ("C", 4)
    ]
    jesus_loves_me_melody = [
        ("G4", 1),
        ("E4", 1),
        ("E4", 1),
        ("D4", 1),
        ("E4", 1),
        ("G4", 1),
        ("G4", 2),
        ("A4", 1),
        ("A4", 1),
        ("C5", 1),
        ("A4", 1),
        ("A4", 1),
        ("G4", 1),
        ("G4", 2),
        ("G4", 1),
        ("E4", 1),
        ("E4", 1),
        ("D4", 1),
        ("E4", 1),
        ("G4", 1),
        ("G4", 2),
        ("A4", 1),
        ("A4", 1),
        ("G4", 1),
        ("C4", 1),
        ("E4", 1),
        ("D4", 1),
        ("C4", 2),
        ("G4", 2),
        ("E4", 1),
        ("G4", 1),
        ("A4", 1),
        ("C5", 3),
        ("G4", 2),
        ("E4", 1),
        ("C4", 1),
        ("E4", 1),
        ("D4", 3),
        ("G4", 2),
        ("E4", 1),
        ("G4", 1),
        ("A4", 1),
        ("C5", 2),
        ("A4", 1),
        ("G4", 1),
        ("C4", 1),
        ("E4", 1),
        ("D4", 1),
        ("C4", 4),
    ]
    weights = {
        "chord_descant_congruence": 0.1,
        "note_variety": 0.1,
        "rhythmic_variety": 0.1,
        "voice_leading": 0.5,
        "functional_harmony": 0.05,
        "counterpoint": 0.15
    }
    assert sum(weights.values()) == 1, "Weights must sum to 1"
    chord_mappings = {
        "C": ["C", "E", "G"],
        "Dm": ["D", "F", "A"],
        "Em": ["E", "G", "B"],
        "F": ["F", "A", "C"],
        "G": ["G", "B", "D"],
        "Am": ["A", "C", "E"],
        "Bdim": ["B", "D", "F"]
    }
    violin_notes = [
        # ("C4", 1),
        # ("D4", 1),
        ("E4", 1),
        ("E4", 2),
        ("E4", 3),
        ("F4", 1),
        ("F4", 2),
        ("G4", 1),
        ("G4", 2),
        ("G4", 3),
        ("G4", 4),
        ("A4", 1),
        ("A4", 2),
        ("B4", 1),
        ("B4", 2),
        ("C5", 1),
        ("C5", 2),
        ("C5", 3),
        ("C5", 4),
        ("D5", 1),
        ("D5", 2),
        ("E5", 1),
        ("E5", 2),
        ("F5", 1),
        ("F5", 2),
        ("G5", 1),
        ("G5", 2)
        # ("A5", 1),
        # ("B5", 1),
        # ("C6", 1)
    ]
    viola_notes = [
        # ("G3", 1),
        # ("A3", 1),
        # ("B3", 1),
        ("C4", 1),
        ("C4", 2),
        ("C4", 3),
        ("C4", 4),
        ("D4", 1),
        ("D4", 2),
        ("E4", 1),
        ("E4", 2),
        ("E4", 3),
        ("F4", 1),
        ("F4", 2),
        ("G4", 1),
        ("G4", 2),
        ("G4", 3),
        ("A4", 1),
        ("A4", 2),
        ("B4", 1),
        ("B4", 2),
        ("C5", 1),
        ("C5", 2),
        ("C5", 3),
        ("C5", 4),
        ("D5", 1),
        ("D5", 2),
        ("E5", 1),
        ("E5", 2)
        # ("F5", 1),
        # ("G5", 1),
        # ("A5", 1),
    ]
    preferred_transitions = {
        "G3": ["A3", "B3", "C4", "D4"],
        "A3": ["G3", "B3", "C4", "D4"],
        "B3": ["G3", "A3", "C4", "D4"],
        "C4": ["G3", "A3", "B3", "C4", "D4", "E4", "G4", "A4"],
        "D4": ["C4", "E4", "F4", "A4"],
        "E4": ["C4", "D4", "F4", "G4"],
        "F4": ["D4", "E4", "G4", "A4"],
        "G4": ["C4", "A4", "B4", "C5"],
        "A4": ["G4", "B4", "C5", "D5"],
        "B4": ["G4", "A4", "C5", "D5"],
        "C5": ["G4", "A4", "B4", "C5", "D5", "E5", "G5", "A5"],
        "D5": ["C5", "E5", "F5"],
        "E5": ["C5", "D5", "F5", "G5"],
        "F5": ["D5", "E5", "G5", "A5"],
        "G5": ["C5", "E5", "F5", "G5", "A5", "B5", "C6"],
        "A5": ["C5", "D5", "F5", "G5", "B5", "C6", "D6"],
        "B5": ["G5", "A5", "C6", "D6"],
        "C6": ["C5", "G5", "A5", "B5", "C6"],
    }

    # Choose Violin or Viola
    instrument = "viola"
    # instrument = "violin"

    # Instantiate objects for generating harmonization
    chord_data = ChordData(jesus_loves_me_chords)
    melody_data = MelodyData(jesus_loves_me_melody)
    assert chord_data.duration == melody_data.duration, "Chord and melody durations must match"
    fitness_evaluator = FitnessEvaluator(
        chord_data=chord_data,
        melody_data=melody_data,
        chord_mappings=chord_mappings,
        notes=violin_notes if instrument == "violin" else viola_notes,
        weights=weights,
        preferred_transitions=preferred_transitions,
    )
    generator = GeneticDescantGenerator(
        chord_data=chord_data,
        notes=violin_notes if instrument == "violin" else viola_notes,
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    # Generate descant with genetic algorithm
    generated_descant = generator.generate(generations=1000)

    # Render to music21 score and show it
    music21_score = create_score(
        generated_descant, jesus_loves_me_melody, jesus_loves_me_chords,
        chord_mappings, instrument=instrument
    )
    music21_score.show()


if __name__ == "__main__":
    main()
