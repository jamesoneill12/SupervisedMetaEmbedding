import torch
import pickle
import numpy as np


# was ='embeddings/similarity_word_embeddings'
def get_training_data(embedding_path, batch_size = 32):
    embedding_dictionary = load_embeddings(embedding_path)
    vectors = []
    for i, (name, words) in enumerate(embedding_dictionary.items()):
        word_v = np.vstack(list(words.values()))
        vectors.append(word_v)
        if i + 1 == len(embedding_dictionary):
            all_vectors = np.hstack(vectors)
            torch_vectors = torch.from_numpy(all_vectors)
            x = torch.utils.data.DataLoader(torch_vectors, batch_size=batch_size, shuffle=False, num_workers=6)
            concat_vector = dict(zip(list(words), all_vectors))
    # return word_vec dictionary (concat_vector) and batched tensors
    return concat_vector, x


def load_embeddings(name):
    root = "C:/Users/jimon/Projects/word-embeddings-benchmarks/models/"
    with open(root + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_embeddings(embeddings, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(embeddings, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    pass