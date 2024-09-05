# Utility functions

import librosa
import numpy as np

def generate_notes(range=(60, 72), durations=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
                   chromatic=False):
    """
    Generate notes from the given range
    
    Parameters:
        range (tuple): The range of notes to generate expressed as
            as a tuple of two MIDI note integers (start, end)
        durations (list): A list of avilable note durations expressed as
            floats relative to a quarter note
        chromatic (bool): If False (default), only include diatonic notes
    
    Returns:
        notes (list): A list of notes expressed as tuples (pitch, duration)
    """
    notes = [(pitch, duration)
        for pitch in list(librosa.midi_to_note(np.arange(range[0], range[1]+1), unicode=False))
        for duration in durations]
    if not chromatic:
        notes = [note for note in notes if note[0][:-1] in ["C", "D", "E", "F", "G", "A", "B"]]
    return notes

def get_weights_from_user():
    """
    Get weights for fitness evaluator functions from the user
    
    Returns:
        weights (dict): A dictionary of weights for each fitness evaluation function,
                        normalized by the sum of all weights
    """
    weights = {}
    # Prompt user to input a float between 0.0 and 10.0 for each feature
    for feature in ["chord_descant_congruence", "pitch_variety", "rhythmic_variety", 
                    "voice_leading", "functional_harmony", "counterpoint"]:
        while True:
            try:
                weight = float(input(f"Enter weight for {feature} (float between 0.0 and 10.0): "))
                if weight < 0.0 or weight > 10.0:
                    raise ValueError
                if weight == 0.0:
                    print("Warning: A weight of 0.0 will result in the corresponding function not being evaluated")
                break
            except ValueError:
                print("Please enter a valid float between 0.0 and 10.0")
        weights[feature] = weight
    # normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0.0:
        for feature in weights:
            weights[feature] /= total_weight
    return weights