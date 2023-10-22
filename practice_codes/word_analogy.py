import random
import numpy as np

print("---------------------- processing -----------------------------------\n")

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
    input_term = input("\nEnter three words separated with space (EXIT to break): ")
    if input_term == 'EXIT' or input_term == 'e':
        break
    
    else:
        # separate input words to a vector
        inpt = input_term.split(' ')

        # calculate the vector length and distance of first two words
        vector_x = vectors[inpt[0]]
        vector_y = vectors[inpt[1]]
        vector_z = np.array((vectors[inpt[2]]))
    
        #print(inpt[0])
        
        # calculate the point from the 3rd word with the vector
        # z + (y − x)
        vector_xy = np.array(vector_y) - np.array(vector_x)
        analogy_vector = vector_z + vector_xy
        # print(len(analogy_vector))
        
    
        # calculate the distances to other words
        vectors_dist = {}
        
        for key, vector in vectors.items():
            if key not in inpt:
                #Euclidean distance
                distance = np.sqrt(np.sum((np.array(analogy_vector) - np.array(vector)) ** 2))
                vectors_dist[key] = distance
        
        # Sort distances (closest to the input based on distance)
        sorted_distances = sorted(vectors_dist.items(), key=lambda x: x[1])
        
        # Set the word and distance from the tuples 
        #closest_key, closest_distance = sorted_distances[0]
        #print(sorted_distances[0])
        #print(sorted_distances[1])
        #print(sorted_distances[10])
        #print("%35s\t\t%f\n" % (closest_key, closest_distance))

        # Print the two (2) most similar words.
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for i, (key, dist) in enumerate(sorted_distances[:2], 1):
            print("%35s\t\t%f\n" % (key, dist))

        

           