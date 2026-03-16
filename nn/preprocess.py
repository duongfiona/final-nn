# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # turning seqs and labels into np arrays (so can use np.random)
    seqs = np.array(seqs)
    labels = np.array(labels)

    # determining where positive and negative sequences are
    pos_idx = np.where(labels)[0]
    neg_idx = np.where(~labels)[0]

    # setting target size to be the size of the majority class
    target_size = max(len(pos_idx), len(neg_idx))

    # sampling minority class WITH replacement to create balanced datasets
    if len(pos_idx) < len(neg_idx):
        sampled_pos_idx = np.random.choice(pos_idx, size=target_size, replace=True)
        sampled_neg_idx = neg_idx
    elif len(neg_idx) < len(pos_idx):
        sampled_neg_idx = np.random.choice(neg_idx, size=target_size, replace=True)
        sampled_pos_idx = pos_idx
    
    sampled_idx = np.concatenate([sampled_pos_idx, sampled_neg_idx])
    np.random.shuffle(sampled_idx) # shuffling to make sure classes are mixed

    # return sampled seqs and labels
    sampled_seqs = seqs[sampled_idx].tolist()
    sampled_labels = labels[sampled_idx].tolist()

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    # creating encodings
    unique_chars = sorted({char for s in seq_arr for char in s})
    encoding_dict = {}
    for i, char in enumerate(unique_chars):
        encoding = np.zeros(len(unique_chars))
        encoding[i] = 1
        encoding_dict[char] = encoding

    encoded_seqs = []
    # applying encodings to each string in seq_arr
    for s in seq_arr:
        # retrieving encodings for each character
        encoded_chars = [encoding_dict[char] for char in s]
        # flattening into 1D arrays
        encoded_s = np.concatenate(encoded_chars)
        encoded_seqs.append(encoded_s)

    # returning all encoded sequences stacked together in an array
    return np.vstack(encoded_seqs)

    
