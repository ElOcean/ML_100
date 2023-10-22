import random
import numpy as np


vocabulary_file='data/word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])
print(vocab['king'])



# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)
    
# Main loop for analogy
while True:
    input_term = input("\nEnter the word (EXIT to break): ")
    if input_term == 'EXIT' or input_term == 'e':
        break
    
    else:
        
        vector_for_key = vectors[input_term]
        #print(vocab[input_term])
        #print(f"Vector for '{ivocab[vocab[input_term]]}': {vector_for_key}")
        #print(f"Sum of the vector: {np.sum(vector_for_key)}")

        vectors_dist = {}
        
        for key, vector in vectors.items():
            #Euclidean distance
            distance = np.sqrt(np.sum((np.array(vector_for_key) - np.array(vector)) ** 2))
            vectors_dist[key] = distance
        
        # Sort distances
        sorted_distances = sorted(vectors_dist.items(), key=lambda x: x[1])
        
        # Find the key closest to the input based on distance
        # closest_key, closest_distance = sorted_distances[0]
        

        #For any input word, return the three (3) most similar words (the most similar should be the input word itself).
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for i, (word, distance) in enumerate(sorted_distances[:3], 1):
            #print(f"{i}. closest word based on distance: {word} (Distance: {distance:.4f})")
            print("%35s\t\t%f\n" % (word, distance))

        
        #print(f"\nClosest word based on distance: '{closest_key}' (Distance: {closest_distance:.4f})")
        #print("%35s\t\t%f\n" % (ivocab[vocab[input_term]], {vectors_dist[input_term]}))
        #for x in a:
            #print("%35s\t\t%f\n" % (ivocab[x], 666))
        

           