import numpy as np
import scipy
import pickle
import torch

BATCH_SIZE = 32

def normalize(y): return ((y-min(y))/(max(y)-min(y)))

def evaluate_similarity(w, X, y):
    mean_vector = np.mean(np.vstack(w.values()), axis=0, keepdims=True)
    A = np.vstack(w[word] if word in list(w.keys()) else mean_vector for word in X[:, 0])
    B = np.vstack(w[word]if word in list(w.keys()) else mean_vector for word in X[:, 1])
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(A, B)])
    return scipy.stats.spearmanr(scores, y).correlation


def get_spearman(x1, x2, y):
    scores = np.array([v1.dot(v2.T)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(x1, x2)])
    return scipy.stats.spearmanr(scores, y).correlation


# was ='embeddings/similarity_word_embeddings'
def get_training_data(embedding_path):
    embedding_dictionary = load_embeddings(embedding_path)
    vectors = []
    for i, (name, words) in enumerate(embedding_dictionary.items()):
        word_v = np.vstack(list(words.values()))
        vectors.append(word_v)
        if i + 1 == len(embedding_dictionary):
            all_vectors = np.hstack(vectors)
            torch_vectors = torch.from_numpy(all_vectors)
            x = torch.utils.data.DataLoader(
                torch_vectors,
                                            batch_size=BATCH_SIZE, shuffle=False, num_workers=6
                                            )
            concat_vector = dict(zip(list(words), all_vectors))
    # return word_vec dictionary (concat_vector) and batched tensors
    return (concat_vector, x)


def load_embeddings(name):
    root = "models/"
    with open(root + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_embeddings(embeddings, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)