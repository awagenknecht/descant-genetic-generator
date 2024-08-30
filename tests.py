import unittest
from unittest.mock import patch
from genetic_descant_generator import MelodyData, ChordData, GeneticDescantGenerator, FitnessEvaluator
import utils

class TestMelodyData(unittest.TestCase):
    def test_melody_data_duration_and_bars(self):
        notes = [("C4", 1), ("D4", 1), ("E4", 2), ("F4", 4), ("B4", 4), ("C4", 4)]
        melody = MelodyData(notes=notes)
        self.assertEqual(melody.duration, 16)
        self.assertEqual(melody.number_of_bars, 4)

class TestChordData(unittest.TestCase):
    def test_chord_data_duration_and_bars(self):
        chords = [("C", 4), ("G", 4), ("Bdim", 4), ("C", 4)]
        chord_data = ChordData(chords=chords)
        self.assertEqual(chord_data.duration, 16)
        self.assertEqual(chord_data.number_of_bars, 4)

class TestGeneticDescantGenerator(unittest.TestCase):
    def setUp(self):
        self.chords = [("C", 4), ("Dm", 4), ("G", 4), ("C", 4)]
        self.chord_data = ChordData(chords=self.chords)
        self.notes = [("C4", 1), ("C4", 2), ("C4", 3), ("C4", 4), 
                      ("D4", 1), ("D4", 2), ("D4", 3), ("D4", 4),
                      ("E4", 1), ("E4", 2), ("E4", 3), ("E4", 4),
                      ("F4", 1), ("F4", 2), ("F4", 3), ("F4", 4),
                      ("G4", 1), ("G4", 2), ("G4", 3), ("G4", 4),
                      ("A4", 1), ("A4", 2), ("A4", 3), ("A4", 4),
                      ("B4", 1), ("B4", 2), ("B4", 3), ("B4", 4)]
        self.melody = [("C4", 4), ("D4", 4), ("D4", 4), ("E4", 4)]
        self.melody_data = MelodyData(notes=self.melody)
        self.fitness_evaluator = FitnessEvaluator(
            chord_data=self.chord_data, 
            melody_data=self.melody_data, 
            chord_mappings={"C": ["C", "E", "G"],
                            "Dm": ["D", "F", "A"],
                            "Em": ["E", "G", "B"],
                            "F": ["F", "A", "C"],
                            "G": ["G", "B", "D"],
                            "Am": ["A", "C", "E"],
                            "Bdim": ["B", "D", "F"]
                            },
            notes=self.notes, 
            weights={"chord_descant_congruence": 0.1,
                    "pitch_variety": 0.1,
                    "rhythmic_variety": 0.1,
                    "voice_leading": 0.5,
                    "functional_harmony": 0.1,
                    "counterpoint": 0.1
                    },
            preferred_transitions={"G3": ["A3", "B3", "C4", "D4"],
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
        )
        self.generator = GeneticDescantGenerator(
            chord_data=self.chord_data, 
            notes=self.notes, 
            population_size=10, 
            mutation_rate=1.0, 
            fitness_evaluator=self.fitness_evaluator
        )
    
    def test_initial_population_size(self):
        population = self.generator._initialise_population()
        self.assertEqual(len(population), 10)
    
    def test_generate_random_notes(self):
        random_note_sequence = self.generator._generate_random_notes()
        self.assertEqual(sum(duration for _, duration in random_note_sequence), 16)

    @patch("random.choice")
    def test_mutation_applies(self, mock_random_choice):
        mock_random_choice.return_value = ("G4", 1)
        note_sequence = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1)]
        mutated_sequence = self.generator._mutate(note_sequence)
        self.assertIn(("G4", 1), mutated_sequence)

    @patch("random.randint")
    def test_crossover(self, mock_randint):
        note_sequence_1 = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1), ("G4", 4), ("A4", 4), ("B4", 4)]
        note_sequence_2 = [("G4", 1), ("A4", 1), ("B4", 1), ("C5", 1), ("G4", 4), ("A4", 4), ("F4", 4)]
        mock_randint.return_value = 2
        crossover_sequence = self.generator._crossover(note_sequence_1, note_sequence_2)
        self.assertEqual(
            crossover_sequence, [("C4", 1), ("D4", 1), ("B4", 1), ("C5", 1), ("G4", 4), ("A4", 4), ("F4", 4)]
            )

    @patch("random.randint")
    def test_crossover_parent1_long(self, mock_randint):
        note_sequence_1 = [("C4", 3), ("F4", 1), ("G4", 4), ("A4", 4), ("B4", 4)]
        note_sequence_2 = [("G4", 1), ("A4", 1), ("B4", 1), ("C5", 1), ("G4", 4), ("A4", 4), ("F4", 4)]
        mock_randint.return_value = 2
        crossover_sequence = self.generator._crossover(note_sequence_1, note_sequence_2)
        self.assertEqual(
            crossover_sequence, [("C4", 2), ("B4", 1), ("C5", 1), ("G4", 4), ("A4", 4), ("F4", 4)]
            )

    @patch("random.randint")
    def test_crossover_parent2_long(self, mock_randint):
        note_sequence_1 = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1), ("G4", 4), ("A4", 4), ("B4", 4)]
        note_sequence_2 = [("C5", 4), ("G4", 4), ("A4", 4), ("F4", 4)]
        mock_randint.return_value = 2
        crossover_sequence = self.generator._crossover(note_sequence_1, note_sequence_2)
        self.assertEqual(
            crossover_sequence, [("C4", 1), ("D4", 1), ("C5", 2), ("G4", 4), ("A4", 4), ("F4", 4)]
            )

    def test_generate(self):
        descant = self.generator.generate(1)
        self.assertEqual(sum(duration for _, duration in descant), 16)

class TestFitnessEvaluator(unittest.TestCase):
    def test_chord_descant_congruence(self):
        chords = [("C", 2), ("G", 2)]
        chord_data = ChordData(chords=chords)
        descant = [("C4", 1), ("G4", 1), ("E4", 1), ("D4", 1)]
        evaluator = FitnessEvaluator(
            chord_data=chord_data, 
            melody_data=None,
            chord_mappings={"C": ["C", "E", "G"], "G": ["G", "B", "D"]},
            notes=[("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1), ("G4", 1)],
            weights={}, 
            preferred_transitions={}
        )
        score = evaluator._chord_descant_congruence(descant)
        self.assertGreater(score, 0)
        self.assertGreater(1, score)
    
    def test_pitch_variety(self):
        notes = [("C4", 1), ("D4", 1), ("E4", 1), ("G4", 1)]
        descant = [("C4", 1), ("C4", 1), ("G4", 1), ("G4", 1)]
        evaluator = FitnessEvaluator(
            chord_data=None, 
            melody_data=None,
            chord_mappings={},
            notes=notes,
            weights={}, 
            preferred_transitions={}
        )
        score = evaluator._pitch_variety(descant)
        self.assertEqual(score, 0.5)

    def test_rhythmic_variety(self):
        notes = [("C4", 1), ("C4", 2), ("C4", 3), ("C4", 4)]
        descant = [("C4", 1), ("C4", 1), ("C4", 2), ("C4", 2)]
        evaluator = FitnessEvaluator(
            chord_data=None, 
            melody_data=None,
            chord_mappings={},
            notes=notes,
            weights={}, 
            preferred_transitions={}
        )
        score = evaluator._rhythmic_variety(descant)
        self.assertEqual(score, 0.5)

    def test_voice_leading(self):
        notes = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1)]
        descant = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1)]
        evaluator = FitnessEvaluator(
            chord_data=None, 
            melody_data=None,
            chord_mappings={},
            notes=notes,
            weights={}, 
            preferred_transitions={"C4":"D4", "D4":"E4", "E4":"F4"}
        )
        score = evaluator._voice_leading(descant)
        self.assertEqual(score, 1)

    def test_functional_harmony(self):
        notes = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1)]
        descant = [("C4", 1), ("D4", 1), ("E4", 1), ("G4", 1)]
        evaluator = FitnessEvaluator(
            chord_data=None, 
            melody_data=None,
            chord_mappings={"C": ["C", "E", "G"], "Dm": ["D", "F", "A"], "Em": ["E", "G", "B"], "F": ["F", "A", "C"]},
            notes=notes,
            weights={}, 
            preferred_transitions={}
        )
        score = evaluator._functional_harmony(descant)
        self.assertEqual(score, 1)

    def test_counterpoint(self):
        notes = [("C4", 1), ("D4", 1), ("E4", 1), ("F4", 1)]
        descant = [("E4", 1), ("F4", 1), ("F4", 1), ("F4", 1)]
        evaluator = FitnessEvaluator(
            chord_data=None, 
            melody_data=MelodyData(notes=notes),
            chord_mappings={},
            notes=notes,
            weights={}, 
            preferred_transitions={}
        )
        score = evaluator._counterpoint(descant)
        self.assertEqual(score, 0.75)


class TestUtils(unittest.TestCase):
    def test_generate_notes(self):
        notes = utils.generate_notes(range=(60, 72), durations=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        self.assertEqual(len(notes), 104)

    def test_get_weights_from_user(self):
        with patch("builtins.input", side_effect=["1", "2", "3", "4", "5", "6"]):
            weights = utils.get_weights_from_user()
        self.assertEqual(weights, {
            "chord_descant_congruence": 1/21,
            "pitch_variety": 2/21,
            "rhythmic_variety": 3/21,
            "voice_leading": 4/21,
            "functional_harmony": 5/21,
            "counterpoint": 6/21
        })

    def test_get_weights_from_user_all_zero(self):
        with patch("builtins.input", side_effect=["0", "0", "0", "0", "0", "0"]):
            weights = utils.get_weights_from_user()
        self.assertEqual(weights, {
            "chord_descant_congruence": 0,
            "pitch_variety": 0,
            "rhythmic_variety": 0,
            "voice_leading": 0,
            "functional_harmony": 0,
            "counterpoint": 0
        })


if __name__ == "__main__":
    unittest.main()
