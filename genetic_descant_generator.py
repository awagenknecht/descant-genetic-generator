import random
from dataclasses import dataclass

import music21

# TODO list:
# 1. Make it work with other durations of notes besides quarter notes
# 2. Allow input of melody along with chords
# 3. Add more evaluation functions
#    - congruence and variety between melody and descant
#    - harmonic functionality of descant
# 4. Make it work with different key and time signatures
# 5. Allow input of XML file with melody and chord voicings
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
        for _ in range(generations):
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
        cut_index = random.randint(1, len(parent1) - 1)
        return parent1[:cut_index] + parent2[cut_index:]

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
    Evaluates the fitness of a chord sequence based on various musical criteria.

    Attributes:
        chords (list): List of tuples representing chords as (chord symbol, duration).
        chord_mappings (dict): Dictionary of chords with their corresponding notes.
        notes (list): List of available notes for the descant.
        weights (dict): Weights for different fitness evaluation functions.
        preferred_transitions (dict): Preferred transitions between notes.
    """

    def __init__(
        self, chord_data, chord_mappings, notes, weights, preferred_transitions
    ):
        """
        Initialize the FitnessEvaluator with accompaniment, chord mappings, notes,
        weights, and preferred transitions.

        Parameters:
            chord_data (ChordData): Accompaniment information.
            chord_mappings (dict): Available chords mapped to their notes.
            notes (list): Available notes for the descant.
            weights (dict): Weights for each fitness evaluation function.
            preferred_transitions (dict): Preferred transitions between notes.
        """
        self.chord_data = chord_data
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

    def _chord_melody_congruence(self, note_sequence):
        """
        Calculates the congruence between the chord sequence and the melody.
        This function assesses how well each chord in the sequence aligns
        with the corresponding segment of the melody. The alignment is
        measured by checking if the notes in the melody are present in the
        chords being played at the same time, rewarding sequences where the
        melody notes fit well with the chords.

        Parameters:
            note_sequence (list): A list of notes to be evaluated against the
                accompaniment.

        Returns:
            float: A score representing the degree of congruence between the
                chord sequence and the melody, normalized by the melody's
                duration.
        """
        score, melody_index = 0, 0
        for chord in self.chord_data.chords:
            bar_duration = 0
            while bar_duration < 4 and melody_index < len(note_sequence):
                pitch, duration = note_sequence[melody_index]
                if pitch[0] in self.chord_mappings[chord[0]]:
                    score += duration
                bar_duration += duration
                melody_index += 1
        return score / self.chord_data.duration

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

    def _melodic_flow(self, note_sequence):
        """
        Assesses the melodic flow of the note sequence by examining the
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
            next_note = note_sequence[i + 1]
            if next_note in self.preferred_transitions[note_sequence[i][0]]:
                score += 1
        return score / (len(note_sequence) - 1)

    # def _functional_harmony(self, chord_sequence):
    #     """
    #     Evaluates the chord sequence based on principles of functional harmony.
    #     This function checks for the presence of key harmonic functions such as
    #     the tonic at the beginning and end of the sequence and the presence of
    #     subdominant and dominant chords. Adherence to these harmonic
    #     conventions is rewarded with a higher score.

    #     Parameters:
    #         chord_sequence (list): The chord sequence to evaluate.

    #     Returns:
    #         float: A score representing the extent to which the sequence
    #             adheres to traditional functional harmony, normalized by
    #             the number of checks performed.
    #     """
    #     score = 0
    #     if chord_sequence[0] in ["C", "Am"]:
    #         score += 1
    #     if chord_sequence[-1] in ["C"]:
    #         score += 1
    #     if "F" in chord_sequence and "G" in chord_sequence:
    #         score += 1
    #     return score / 3


def create_score(descant, chord_sequence, chord_mappings):
    """
    Create a music21 score with a given descant and chord sequence.

    Args:
        descant (list): A list of tuples representing notes in the format
            (note_name, duration).
        chord_sequence (list): A list of tuples representing chords
            in the format (chord symbol, duration).

    Returns:
        music21.stream.Score: A music score containing the descant and chord
            sequence.
    """
    # Create a Score object
    score = music21.stream.Score()

    # Create the descant part and add notes to it
    descant_part = music21.stream.Part()
    for note_name, duration in descant:
        descant_note = music21.note.Note(note_name, quarterLength=duration)
        descant_part.append(descant_note)

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
    score.append(chord_part)

    return score


def main():

    # twinkle_twinkle_melody = [
    #     ("C5", 1),
    #     ("C5", 1),
    #     ("G5", 1),
    #     ("G5", 1),
    #     ("A5", 1),
    #     ("A5", 1),
    #     ("G5", 2),  # Twinkle, twinkle, little star,
    #     ("F5", 1),
    #     ("F5", 1),
    #     ("E5", 1),
    #     ("E5", 1),
    #     ("D5", 1),
    #     ("D5", 1),
    #     ("C5", 2),  # How I wonder what you are!
    #     ("G5", 1),
    #     ("G5", 1),
    #     ("F5", 1),
    #     ("F5", 1),
    #     ("E5", 1),
    #     ("E5", 1),
    #     ("D5", 2),  # Up above the world so high,
    #     ("G5", 1),
    #     ("G5", 1),
    #     ("F5", 1),
    #     ("F5", 1),
    #     ("E5", 1),
    #     ("E5", 1),
    #     ("D5", 2),  # Like a diamond in the sky.
    #     ("C5", 1),
    #     ("C5", 1),
    #     ("G5", 1),
    #     ("G5", 1),
    #     ("A5", 1),
    #     ("A5", 1),
    #     ("G5", 2),  # Twinkle, twinkle, little star,
    #     ("F5", 1),
    #     ("F5", 1),
    #     ("E5", 1),
    #     ("E5", 1),
    #     ("D5", 1),
    #     ("D5", 1),
    #     ("C5", 2)  # How I wonder what you are!
    # ]
    
    # weights = {
    #     "chord_melody_congruence": 0.4,
    #     "chord_variety": 0.1,
    #     "harmonic_flow": 0.3,
    #     "functional_harmony": 0.2
    # }

    # preferred_transitions = {
    #     "C": ["G", "Am", "F"],
    #     "Dm": ["G", "Am"],
    #     "Em": ["Am", "F", "C"],
    #     "F": ["C", "G"],
    #     "G": ["Am", "C"],
    #     "Am": ["Dm", "Em", "F", "C"],
    #     "Bdim": ["F", "Am"]
    # }

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
    weights = {
        "chord_melody_congruence": 0.4,
        "note_variety": 0.1,
        "melodic_flow": 0.3,
    }
    chord_mappings = {
        "C": ["C", "E", "G"],
        "Dm": ["D", "F", "A"],
        "Em": ["E", "G", "B"],
        "F": ["F", "A", "C"],
        "G": ["G", "B", "D"],
        "Am": ["A", "C", "E"],
        "Bdim": ["B", "D", "F"]
    }
    notes = [
        ("C5", 1),
        # ("C5", 2),
        # ("C5", 3),
        # ("C5", 4),
        ("D5", 1),
        # ("D5", 2),
        ("E5", 1),
        # ("E5", 2),
        # ("E5", 4),
        ("F5", 1),
        # ("F5", 2),
        ("G5", 1),
        # ("G5", 2),
        # ("G5", 3),
        # ("G5", 4),
        ("A5", 1),
        # ("A5", 2),
        # ("A5", 3),
        # ("A5", 4),
        ("B5", 1),
        # ("B5", 2),
        ("C6", 1),
        # ("C6", 2),
        # ("C6", 3),
        # ("C6", 4),
        ("D6", 1),
        # ("D6", 2),
        ("E6", 1),
        # ("E6", 2),
        # ("E6", 4),
    ]
    preferred_transitions = {
        "C5": ["C5", "D5", "E5", "G5", "A5"],
        "D5": ["C5", "E5", "F5"],
        "E5": ["C5", "D5", "F5", "G5"],
        "F5": ["D5", "E5", "G5", "A5"],
        "G5": ["C5", "E5", "F5", "G5", "A5", "B5", "C6"],
        "A5": ["C5", "D5", "F5", "G5", "B5", "C6", "D6"],
        "B5": ["G5", "A5", "C6", "D6"],
        "C6": ["G5", "B5", "C6", "D6", "E6"],
        "D6": ["C6", "E6"],
        "E6": ["C6", "D6"],
    }

    # Instantiate objects for generating harmonization
    chord_data = ChordData(jesus_loves_me_chords)
    fitness_evaluator = FitnessEvaluator(
        chord_data=chord_data,
        chord_mappings=chord_mappings,
        notes=notes,
        weights=weights,
        preferred_transitions=preferred_transitions,
    )
    generator = GeneticDescantGenerator(
        chord_data=chord_data,
        notes=notes,
        population_size=100,
        mutation_rate=0.05,
        fitness_evaluator=fitness_evaluator,
    )

    # Generate descant with genetic algorithm
    generated_descant = generator.generate(generations=1000)

    # Render to music21 score and show it
    music21_score = create_score(
        generated_descant, jesus_loves_me_chords, chord_mappings
    )
    music21_score.show()


if __name__ == "__main__":
    main()
