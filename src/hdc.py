# Library for HD vector operations
import numpy as np

def generate_item_memory(characters, dimension, seed=None):
    """
    Random hypervector generation for Item Memory (IM) creation
    
    Parameters:
        characters (list or str): The alphabet/symbols to encode (26 letters + space).
        dimension (int): The length of the hypervectors (D).
        seed (int, optional): Random seed for reproducibility across sweeps.
        
    Returns:
        dict: A lookup dictionary mapping characters to their binary NumPy arrays.
    """
    if seed is not None:
        np.random.seed(seed)
        
    item_memory = {}
    for char in characters:
        # Generates an array of random 0s and 1s of size (dimension,)
        item_memory[char] = np.random.randint(2, size=dimension, dtype=np.uint8)
        
    return item_memory


def bind(hv1, hv2):
    """
    Bitwise XOR.
    
    Parameters:
        hv1, hv2 (NumPy arrays): Binary hypervectors of the same dimension.
        
    Returns:
        NumPy array: The bound, completely orthogonal binary hypervector.
    """
    return np.bitwise_xor(hv1, hv2)


def permute(hv, steps=1):
    """
    Cyclic shift.
    
    Parameters:
        hv (NumPy array): The binary hypervector to permute.
        steps (int): Number of positions to shift right (default is 1).
        
    Returns:
        NumPy array: The permuted binary hypervector.
    """
    return np.roll(hv, shift=steps)

# No separate bundle function; accumulator/majority gate handled in encoder.py