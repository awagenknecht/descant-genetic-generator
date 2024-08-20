# descant-genetic-generator
Generates a violin or viola descant for an input song using a genetic algorithm.

The genetic algorithm starts with a population of randomly generated note sequences and iteratively improves them. At each iteration, a new population of children note sequences is generated by crossover between parents plus random mutations. The best note sequences are selected by custom fitness functions that evaluate the descant in relation to the song melody and chords as well as the descant's voice leading.

Inspiration comes from Valerio Velardo's course on Generative Music AI. Valerio's code for harmonization generation was borrowed from the following repository and repurposed for descant generation:
https://github.com/musikalkemist/generativemusicaicourse
