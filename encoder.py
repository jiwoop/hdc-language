import numpy as np
import hdc

def encode_text_to_hv(text, item_memory, n_gram_size, dimension, max_val=None):
    # Init accumulator
    accumulator = np.zeros(dimension, dtype=np.int32)

    # Slide through the text
    for i in range(len(text) - n_gram_size + 1):
        window = text[i:i + n_gram_size]

        # Check if all characters in the window exist in Item Memory
        if any(char not in item_memory for char in window):
            continue

        # Construct the single N-gram hypervector using bind and permute
        # Example for N=3: bind(bind(V_0, permute(V_1, 1)), permute(V_2, 2))
        ngram_hv = item_memory[window[0]]
        for pos in range(1, n_gram_size):
            permuted_hv = hdc.permute(item_memory[window[pos]], steps=pos)
            ngram_hv = hdc.bind(ngram_hv, permuted_hv)

        # Stream straight into the accumulator (Map 1 -> +1, 0 -> -1)
        bipolar_vector = 2 * ngram_hv.astype(np.int32) - 1
        accumulator += bipolar_vector

        # Saturating counter: clamp to [-max_val, max_val] each step, as a
        # fixed-width HW accumulator would, instead of growing unbounded.
        if max_val is not None:
            np.clip(accumulator, -max_val, max_val, out=accumulator)

    # Final Majority Gate (This replaces the final step of the bundle function!)
    # Elements > 0 get a 1, elements <= 0 get a 0
    final_class_hv = (accumulator > 0).astype(np.uint8)

    return final_class_hv