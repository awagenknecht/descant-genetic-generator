# Utility functions

import librosa
import numpy as np

def generate_notes(range=(60, 72), durations=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]):
    """
    Generate notes from the given range
    
    Parameters:
        range (tuple): The range of notes to generate expressed as
            as a tuple of two MIDI note integers (start, end)
        durations (list): A list of avilable note durations expressed as
            floats relative to a quarter note
    
    Returns:
        notes (list): A list of notes expressed as tuples (pitch, duration)
    """
    return [(pitch, duration)
        for pitch in list(librosa.midi_to_note(np.arange(range[0], range[1]+1)))
        for duration in durations
        ]