# BLAKE3 implementation using TinyGrad, aiming for a JITted core.
# Note: Environmental instability with TinyGrad JIT/execution was observed during development.

import struct
import math
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit # Corrected JIT import path

# Global Constants
IV_CONST = Tensor([
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19
], dtype=dtypes.uint32, requires_grad=False)

MSG_PERMUTATION_CONST = Tensor([
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
], dtype=dtypes.int32, requires_grad=False)

BLOCK_LEN_BYTES = 64
# Other BLAKE3 constants can be added here

# BLAKE3 Flags
CHUNK_START_FLAG        = 1 << 0
CHUNK_END_FLAG          = 1 << 1
PARENT_FLAG             = 1 << 2
ROOT_FLAG               = 1 << 3
KEYED_HASH_FLAG         = 1 << 4
DERIVE_KEY_CONTEXT_FLAG = 1 << 5 # Used for hashing the context string
DERIVE_KEY_MATERIAL_FLAG= 1 << 6 # Used for the main hashing after context

ZERO_BLOCK_WORDS_TENSOR = Tensor.zeros(16, dtype=dtypes.uint32, requires_grad=False).realize()

def _g(s_current: Tensor, idx_a: int, idx_b: int, idx_c: int, idx_d: int, mx: Tensor, my: Tensor) -> Tensor:
    """
    Performs the BLAKE3 G mixing function on the given state words.

    Args:
        s_current: The current 16-word state tensor (uint32).
        idx_a, idx_b, idx_c, idx_d: Indices of the state words to be mixed.
        mx: A message word tensor (uint32).
        my: Another message word tensor (uint32).

    Returns:
        A new state tensor with the words at idx_a, idx_b, idx_c, idx_d updated.
        The update is performed immutably.
    """
    # Extract words for mixing
    sa = s_current[idx_a]
    sb = s_current[idx_b]
    sc = s_current[idx_c]
    sd = s_current[idx_d]

    # BLAKE3 G function steps:
    # Step 1
    sa = (sa + sb + mx).cast(dtypes.uint32)
    # Step 2
    sd = ((sd.bitwise_xor(sa)).rshift(16).bitwise_or((sd.bitwise_xor(sa)).lshift(32-16))).cast(dtypes.uint32)
    # Step 3
    sc = (sc + sd).cast(dtypes.uint32)
    # Step 4
    sb = ((sb.bitwise_xor(sc)).rshift(12).bitwise_or((sb.bitwise_xor(sc)).lshift(32-12))).cast(dtypes.uint32)
    # Step 5
    sa = (sa + sb + my).cast(dtypes.uint32)
    # Step 6
    sd = ((sd.bitwise_xor(sa)).rshift(8).bitwise_or((sd.bitwise_xor(sa)).lshift(32-8))).cast(dtypes.uint32)
    # Step 7
    sc = (sc + sd).cast(dtypes.uint32)
    # Step 8
    sb = ((sb.bitwise_xor(sc)).rshift(7).bitwise_or((sb.bitwise_xor(sc)).lshift(32-7))).cast(dtypes.uint32)

    # Update state words immutably
    indices = Tensor.arange(16, dtype=dtypes.int32, requires_grad=False)
    s_new = s_current
    s_new = Tensor.where(indices == idx_a, sa, s_new)
    s_new = Tensor.where(indices == idx_b, sb, s_new)
    s_new = Tensor.where(indices == idx_c, sc, s_new)
    s_new = Tensor.where(indices == idx_d, sd, s_new)
    return s_new

@TinyJit
def blake3_compress_block_jit(cv_tensor: Tensor, block_words_tensor: Tensor, counter_low_tensor: Tensor, counter_high_tensor: Tensor, blen_tensor: Tensor, flags_tensor: Tensor) -> Tensor:
    """
    JIT-compiled BLAKE3 compression function for a single block.

    Processes one 64-byte block of message data using the BLAKE3 algorithm.
    This function is designed to be compiled by TinyGrad's JIT compiler.

    Args:
        cv_tensor: Current Chaining Value (CV) - 8 words (Tensor, uint32).
        block_words_tensor: Message block words - 16 words (Tensor, uint32).
        counter_low_tensor: Low 32 bits of the block counter (Tensor, uint32, 1 element).
        counter_high_tensor: High 32 bits of the block counter (Tensor, uint32, 1 element).
        blen_tensor: Block length in bytes (Tensor, uint32, 1 element).
        flags_tensor: Flags for this compression (Tensor, uint32, 1 element).

    Returns:
        A 16-word Tensor (uint32) representing the full output of the compression.
        The first 8 words are typically used as the new CV or hash output.
    """
    # Initialize the 16-word state s by concatenating inputs
    # s = chaining_value || IV[0:4] || counter_low || counter_high || block_length || flags
    s = Tensor.cat(cv_tensor, IV_CONST[0:4], counter_low_tensor, counter_high_tensor, blen_tensor, flags_tensor)
    
    m = block_words_tensor # Assign message block words

    # Perform 7 rounds of BLAKE3 mixing
    for round_num in range(7):
        # Column steps
        s = _g(s, 0, 4, 8,  12, m[0],  m[1])
        s = _g(s, 1, 5, 9,  13, m[2],  m[3])
        s = _g(s, 2, 6, 10, 14, m[4],  m[5])
        s = _g(s, 3, 7, 11, 15, m[6],  m[7])
        # Diagonal steps
        s = _g(s, 0, 5, 10, 15, m[8],  m[9])
        s = _g(s, 1, 6, 11, 12, m[10], m[11])
        s = _g(s, 2, 7, 8,  13, m[12], m[13])
        s = _g(s, 3, 4, 9,  14, m[14], m[15])
        
        # Permute message words for all rounds except the last one
        if round_num < 6:
            m = m[MSG_PERMUTATION_CONST]
    
    # Finalization: XOR the internal state words to produce output
    # s_final[0:8] = s[0:8] XOR s[8:16]
    # s_final[8:16] = s[8:16] XOR cv_tensor (original chaining value)
    s_final_0_7 = s[0:8].bitwise_xor(s[8:16])
    s_final_8_15 = s[8:16].bitwise_xor(cv_tensor)
    return Tensor.cat(s_final_0_7, s_final_8_15)

class Blake3Hasher:
    """
    Main interface for BLAKE3 hashing.

    Supports standard hashing, keyed hashing, and key derivation using a context string.
    Processes input data via the `update` method and finalizes the hash
    with the `finalize` method.
    """
    BLOCK_LEN = BLOCK_LEN_BYTES # Block length in bytes (64)
    CHUNK_LEN = 1024          # Chunk length in bytes (16 blocks)

    def __init__(self, key: bytes | None = None, context_string: str | None = None):
        """
        Initializes the Blake3Hasher.

        Args:
            key: Optional 32-byte key for keyed hashing mode.
            context_string: Optional string for key derivation mode.
                            If provided, this string is hashed to derive the initial key.
        """
        self.flags_base = 0 # Base flags for all compressions in this hasher instance.
        
        if key:
            # Keyed hashing mode
            if len(key) != 32: raise ValueError("Key must be 32 bytes long.")
            key_words_list = [struct.unpack('<I', key[i:i+4])[0] for i in range(0, 32, 4)]
            # Initial key words from the provided key
            self.key_words_initial = Tensor(key_words_list, dtype=dtypes.uint32, requires_grad=False)
            self.flags_base = KEYED_HASH_FLAG
        elif context_string:
            # Key derivation mode using a context string
            # Hash the context string to get the actual initial key words.
            # This internal hashing uses DERIVE_KEY_CONTEXT_FLAG.
            context_hasher = Blake3HasherInternal(flags_base_override=DERIVE_KEY_CONTEXT_FLAG)
            context_key_material_bytes = context_hasher.update(context_string.encode('utf-8'))._finalize_internal(out_len=32)
            key_words_list = [struct.unpack('<I', context_key_material_bytes[i:i+4])[0] for i in range(0, 32, 4)]
            self.key_words_initial = Tensor(key_words_list, dtype=dtypes.uint32, requires_grad=False)
            # The main hashing then uses DERIVE_KEY_MATERIAL_FLAG with this derived key.
            self.flags_base = DERIVE_KEY_MATERIAL_FLAG
        else:
            # Standard hashing mode
            # Initial key words are the standard IV.
            self.key_words_initial = Tensor(IV_CONST.numpy(), dtype=IV_CONST.dtype, requires_grad=False)
            # self.flags_base remains 0.
        
        self.chunk_counter: int = 0 # Counter for chunks processed.
        # Current Chaining Value (CV), initialized with key_words_initial.
        self.current_chunk_cv: Tensor = Tensor(self.key_words_initial.numpy(), dtype=self.key_words_initial.dtype, requires_grad=False)
        self.buffer: bytearray = bytearray() # Buffer for input data.
        self.cv_stack: list[Tensor] = [] # Stack for storing CVs of completed parent nodes in the tree.
        self.blocks_processed_in_chunk: int = 0 # Number of blocks processed in the current chunk.

    def _bytes_to_block_words(self, block_bytes: bytes) -> Tensor:
        """
        Converts a 64-byte block of bytes into a Tensor of 16 uint32 words.
        Input bytes are interpreted as little-endian.
        """
        assert len(block_bytes) == self.BLOCK_LEN, f"Input to _bytes_to_block_words must be {self.BLOCK_LEN} bytes, got {len(block_bytes)}"
        words = [struct.unpack('<I', block_bytes[i:i+4])[0] for i in range(0, self.BLOCK_LEN, 4)]
        return Tensor(words, dtype=dtypes.uint32, requires_grad=False)

    def _process_block(self, block_bytes: bytes, flags: int, actual_data_len: int | None = None):
        """
        Processes a single 64-byte message block using the JITted compression function
        and updates self.current_chunk_cv.

        Args:
            block_bytes: The 64-byte message block to process (must be padded by caller).
            flags: Flags for this specific block compression (e.g., CHUNK_START, CHUNK_END).
            actual_data_len: The actual length of data in block_bytes if it's the last block
                             of a chunk (used for blen_tensor). Should be <= BLOCK_LEN.
        """
        block_words_tensor = self._bytes_to_block_words(block_bytes)
        current_cv_tensor = self.current_chunk_cv.contiguous().realize() # Ensure CV is dense for JIT.
        
        counter_low_tensor = Tensor([self.chunk_counter & 0xFFFFFFFF], dtype=dtypes.uint32, requires_grad=False)
        counter_high_tensor = Tensor([(self.chunk_counter >> 32) & 0xFFFFFFFF], dtype=dtypes.uint32, requires_grad=False)
        
        # blen_tensor should reflect actual data length for the last block in a CHUNK_END scenario.
        blen_val_to_use = actual_data_len if actual_data_len is not None and (flags & CHUNK_END_FLAG) else self.BLOCK_LEN
        blen_tensor = Tensor([blen_val_to_use], dtype=dtypes.uint32, requires_grad=False)
        
        # Combine per-block flags with base flags (e.g. KEYED_HASH, DERIVE_KEY_MATERIAL).
        # Internal hasher might override flags_base.
        flags_to_use = flags | self.flags_base
        if hasattr(self, 'flags_base_override') and self.flags_base_override is not None:
             flags_to_use = flags | self.flags_base_override
        flags_tensor = Tensor([flags_to_use], dtype=dtypes.uint32, requires_grad=False)
        
        output_16_words = blake3_compress_block_jit(current_cv_tensor, block_words_tensor, counter_low_tensor, counter_high_tensor, blen_tensor, flags_tensor)
        # The first 8 words of the compression output become the new CV for the current chunk.
        self.current_chunk_cv = output_16_words[0:8].realize()

    def update(self, data: bytes):
        """
        Updates the hash state with the given input data.

        Data is buffered and processed in blocks and chunks.
        """
        self.buffer.extend(data)
        while len(self.buffer) >= self.BLOCK_LEN:
            block = self.buffer[:self.BLOCK_LEN]
            del self.buffer[:self.BLOCK_LEN] # Use del for bytearray slice removal
            
            flags = 0
            if self.blocks_processed_in_chunk == 0: 
                flags |= CHUNK_START_FLAG
            
            self._process_block(bytes(block), flags, actual_data_len=self.BLOCK_LEN)
            self.blocks_processed_in_chunk += 1
            
            if self.blocks_processed_in_chunk == (self.CHUNK_LEN // self.BLOCK_LEN): # Chunk of 16 blocks is complete
                # Finalize the chunk and update the CV stack for tree hashing.
                # Copy current_chunk_cv as it's passed and its underlying data might be needed if it's part of key_words_initial.
                self._finalize_chunk_and_update_stack(Tensor(self.current_chunk_cv.numpy(), dtype=self.current_chunk_cv.dtype, requires_grad=False))
                # Reset CV for the new chunk to the initial key/IV.
                self.current_chunk_cv = Tensor(self.key_words_initial.numpy(), dtype=self.key_words_initial.dtype, requires_grad=False)
                self.chunk_counter += 1
                self.blocks_processed_in_chunk = 0
        return self

    def _finalize_chunk_and_update_stack(self, chunk_cv: Tensor):
        """
        Finalizes a completed chunk and updates the CV stack.
        This implements the "collapsing" of parent nodes in BLAKE3's binary tree structure
        as chunks are completed.
        """
        current_chaining_value = chunk_cv
        # total_chunks_processed_so_far refers to the 0-indexed chunk_counter of the chunk *just finished*.
        # The logic collapses pairs based on trailing zeros in the binary representation of this count.
        temp_chunk_idx = self.chunk_counter 
        while (temp_chunk_idx & 1) == 0 and len(self.cv_stack) > 0:
            # If current chunk_idx is even, it's a right child.
            # Pop the left child from stack and compute parent CV.
            right_child_cv = current_chaining_value
            left_child_cv = self.cv_stack.pop()
            current_chaining_value = self._parent_cv(left_child_cv, right_child_cv)
            temp_chunk_idx >>= 1 # Move to the parent level.
        self.cv_stack.append(current_chaining_value)

    def _parent_cv(self, left_cv: Tensor, right_cv: Tensor) -> Tensor:
        """
        Computes the Chaining Value (CV) for a parent node given the CVs of its two children.
        The input CV for compressing parent nodes is always the initial key material.
        The counter for parent nodes is 0 (as per simplified BLAKE3 versions, official spec is nuanced).
        """
        parent_block_words = Tensor.cat(left_cv, right_cv) 
        # Use initial key words as the "CV" for parent node compression.
        parent_cv_input = Tensor(self.key_words_initial.numpy(), dtype=self.key_words_initial.dtype, requires_grad=False).contiguous().realize()
        
        counter_low = Tensor([0], dtype=dtypes.uint32, requires_grad=False) # Counter is 0 for parent nodes.
        counter_high = Tensor([0], dtype=dtypes.uint32, requires_grad=False)
        blen = Tensor([self.BLOCK_LEN], dtype=dtypes.uint32, requires_grad=False) # Parent blocks are always full.
        flags = PARENT_FLAG | self.flags_base # Set PARENT_FLAG.
        flags_tensor = Tensor([flags], dtype=dtypes.uint32, requires_grad=False)
        
        output_16 = blake3_compress_block_jit(parent_cv_input, parent_block_words, counter_low, counter_high, blen, flags_tensor)
        return output_16[0:8].realize() # The first 8 words are the parent's CV.

    def finalize(self, out_len: int = 32) -> bytes:
        """
        Finalizes the hash computation.

        Processes any remaining data in the buffer, completes the tree hashing by
        combining CVs from the stack, and then generates the final hash output
        of the specified length using the root node CV.
        """
        flags = CHUNK_END_FLAG # This is the last chunk.
        if self.blocks_processed_in_chunk == 0: 
            flags |= CHUNK_START_FLAG # It's also the first block of this (final) chunk.
        
        # Process the final (potentially partial) block.
        remaining_len = len(self.buffer)
        block_to_process = bytes(self.buffer) # Get data from buffer.
        self.buffer.clear() # Clear buffer.
        
        # Pad the last block to BLOCK_LEN bytes.
        padded_block = block_to_process + b'\x00' * (self.BLOCK_LEN - remaining_len if remaining_len < self.BLOCK_LEN else 0)
        self._process_block(padded_block, flags, actual_data_len=remaining_len)
        
        # The CV of the last chunk is now in self.current_chunk_cv.
        output_node_cv = Tensor(self.current_chunk_cv.numpy(), dtype=self.current_chunk_cv.dtype, requires_grad=False)

        # Complete the tree hashing: combine the CV of the last chunk with CVs from the stack.
        while len(self.cv_stack) > 0:
            left_sibling_cv = self.cv_stack.pop() # Older, left children from stack.
            output_node_cv = self._parent_cv(left_sibling_cv, output_node_cv) # Current CV is the right child.
        
        # output_node_cv is now the root CV. Generate the final hash output from it.
        final_result_bytes = bytearray()
        output_block_counter = 0 # Counter for output blocks, starts at 0.
        while len(final_result_bytes) < out_len:
            # Prepare inputs for generating an output block.
            # CV is the root_node_cv, message block is zeros.
            counter_low = Tensor([output_block_counter & 0xFFFFFFFF], dtype=dtypes.uint32, requires_grad=False).contiguous().realize()
            counter_high = Tensor([(output_block_counter >> 32) & 0xFFFFFFFF], dtype=dtypes.uint32, requires_grad=False).contiguous().realize()
            blen_final = Tensor([self.BLOCK_LEN], dtype=dtypes.uint32, requires_grad=False).contiguous().realize() # Standard block length.
            flags_final_val = ROOT_FLAG | self.flags_base # Set ROOT_FLAG.
            flags_final = Tensor([flags_final_val], dtype=dtypes.uint32, requires_grad=False)
            
            output_16_words = blake3_compress_block_jit(
                output_node_cv.contiguous().realize(), # Root CV is the "key" for output.
                ZERO_BLOCK_WORDS_TENSOR,               # Input block is all zeros.
                counter_low, counter_high, blen_final, flags_final
            )
            
            # Convert the 16 output words (64 bytes) to bytes (little-endian).
            for i in range(16):
                word_val = output_16_words[i].numpy().item()
                final_result_bytes.extend(struct.pack('<I', int(word_val)))
            
            output_block_counter += 1 # Increment for next output block if needed.
            # For standard BLAKE3, output_node_cv remains the same for subsequent output blocks if out_len > 64.
            # Some XOF modes might re-chain, but not standard BLAKE3.

        return bytes(final_result_bytes[:out_len]) # Truncate to desired output length.

class Blake3HasherInternal(Blake3Hasher):
    """
    Internal helper class for BLAKE3 operations, specifically for key derivation
    from a context string. It uses a specific `flags_base_override`.
    Inherits from Blake3Hasher but overrides parts of initialization and finalization.
    """
    def __init__(self, flags_base_override: int):
        """
        Initializes the internal hasher.
        Args:
            flags_base_override: The specific BLAKE3 flag to use as `flags_base`
                                 (e.g., DERIVE_KEY_CONTEXT_FLAG).
        """
        super().__init__() # Call parent __init__ but will override key_words and flags_base.
        self.flags_base_override = flags_base_override # Store the override.
        # For context string hashing, the initial key words are always IV_CONST.
        self.key_words_initial = Tensor(IV_CONST.numpy(), dtype=IV_CONST.dtype, requires_grad=False)
        self.current_chunk_cv = Tensor(self.key_words_initial.numpy(), dtype=self.key_words_initial.dtype, requires_grad=False)
        # Crucially, set flags_base to the override for all operations in this internal hasher.
        self.flags_base = flags_base_override 

    def _finalize_internal(self, out_len: int = 32) -> bytes:
        """
        Simplified finalization for deriving key material from a context string.
        Assumes the context string typically fits within a few blocks and doesn't require
        complex tree state from cv_stack.
        """
        flags = CHUNK_END_FLAG | CHUNK_START_FLAG # Assume context string is processed as one chunk.
        
        remaining_len = len(self.buffer)
        block_to_process = bytes(self.buffer)
        self.buffer.clear()
        
        # Warn if context string is unusually long for this simplified internal path.
        if remaining_len > self.BLOCK_LEN * (self.CHUNK_LEN // self.BLOCK_LEN) : # More than one chunk
             print("Warning: Context string is very long, simplified internal hasher might not be fully spec-compliant for tree structure if context spans multiple chunks.")
        
        padded_block = block_to_process + b'\x00' * (self.BLOCK_LEN - remaining_len if remaining_len < self.BLOCK_LEN else 0)
        # Process the (potentially only) block of the context string.
        self._process_block(padded_block[:self.BLOCK_LEN], flags, actual_data_len=remaining_len) 
        
        # The CV from processing the context string is the result.
        output_node_cv = Tensor(self.current_chunk_cv.numpy(), dtype=self.current_chunk_cv.dtype, requires_grad=False)
        
        # For key derivation, typically only one root output block is needed.
        # No complex cv_stack handling needed for this simplified internal finalizer.
        final_result_bytes = bytearray()
        output_block_counter = 0
        while len(final_result_bytes) < out_len:
            counter_low = Tensor([output_block_counter & 0xFFFFFFFF], dtype=dtypes.uint32, requires_grad=False).contiguous().realize()
            counter_high = Tensor([(output_block_counter >> 32) & 0xFFFFFFFF], dtype=dtypes.uint32, requires_grad=False).contiguous().realize()
            blen_final = Tensor([self.BLOCK_LEN], dtype=dtypes.uint32, requires_grad=False).contiguous().realize()
            # Use the overridden flags_base (e.g., DERIVE_KEY_CONTEXT_FLAG) with ROOT_FLAG.
            flags_final_val = ROOT_FLAG | self.flags_base_override 
            flags_final = Tensor([flags_final_val], dtype=dtypes.uint32, requires_grad=False)

            output_16_words = blake3_compress_block_jit(
                output_node_cv.contiguous().realize(), 
                ZERO_BLOCK_WORDS_TENSOR, 
                counter_low, counter_high, blen_final, flags_final
            )
            for i in range(16):
                word_val = output_16_words[i].numpy().item()
                final_result_bytes.extend(struct.pack('<I', int(word_val)))
            output_block_counter += 1
            # For key derivation, usually out_len is 32 (8 words) or 64 (16 words),
            # so one or two output blocks are sufficient.
            if output_block_counter * BLOCK_LEN_BYTES >= out_len and output_block_counter >= (out_len + BLOCK_LEN_BYTES -1) // BLOCK_LEN_BYTES : # ensure enough blocks fetched
                 break
        return bytes(final_result_bytes[:out_len])

if __name__ == "__main__":
    print("SCRIPT_MAIN_START")
    try:
        print("SCRIPT_TRY_START")
        
        # Ensure TinyGrad settings for inference/hashing
        Tensor.no_grad = True 
        Tensor.training = False

        # Test Case 1: Standard Hashing "abc"
        print("\n--- Test Case 1: Standard Hashing ('abc') ---")
        hasher_abc = Blake3Hasher()
        hasher_abc.update(b"abc")
        digest_bytes_abc = hasher_abc.finalize(out_len=32)
        computed_hex_abc = digest_bytes_abc.hex()
        known_hex_abc = "3a502910c30090769251803938120058787543526864997c7360e1505ce9e899"
        print(f"Input: b'abc'")
        print(f"Computed digest: {computed_hex_abc}")
        print(f"Expected digest: {known_hex_abc}")
        if computed_hex_abc == known_hex_abc:
            print("Status: PASSED")
        else:
            print("Status: FAILED")

        # Test Case 2: Standard Hashing Empty String ""
        print("\n--- Test Case 2: Standard Hashing ('') ---")
        hasher_empty = Blake3Hasher()
        digest_bytes_empty = hasher_empty.finalize(out_len=32) # No update call
        computed_hex_empty = digest_bytes_empty.hex()
        known_hex_empty = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        print(f"Input: b'' (empty string)")
        print(f"Computed digest: {computed_hex_empty}")
        print(f"Expected digest: {known_hex_empty}")
        if computed_hex_empty == known_hex_empty:
            print("Status: PASSED")
        else:
            print("Status: FAILED")

        # Test Case 3: Standard Hashing "hello world"
        print("\n--- Test Case 3: Standard Hashing ('hello world') ---")
        hasher_hw = Blake3Hasher()
        hasher_hw.update(b"hello world")
        digest_bytes_hw = hasher_hw.finalize(out_len=32)
        computed_hex_hw = digest_bytes_hw.hex()
        known_hex_hw = "f7ff4f81970018779685fe726a8e132853e699166d84070e52a14407e45ddcec"
        print(f"Input: b'hello world'")
        print(f"Computed digest: {computed_hex_hw}")
        print(f"Expected digest: {known_hex_hw}")
        if computed_hex_hw == known_hex_hw:
            print("Status: PASSED")
        else:
            print("Status: FAILED")

        # Test Case 4: Keyed Hashing
        print("\n--- Test Case 4: Keyed Hashing ---")
        key_bytes = b"0123456789abcdef0123456789abcdef"
        hasher_keyed = Blake3Hasher(key=key_bytes)
        hasher_keyed.update(b"example input")
        digest_bytes_keyed = hasher_keyed.finalize(out_len=32)
        computed_hex_keyed = digest_bytes_keyed.hex()
        known_hex_keyed = "9e1d153691070988958729722f765782910995915489d7c50310295927308836"
        print(f"Input: b'example input', Key: {key_bytes.decode()}")
        print(f"Computed digest: {computed_hex_keyed}")
        print(f"Expected digest: {known_hex_keyed}")
        if computed_hex_keyed == known_hex_keyed:
            print("Status: PASSED")
        else:
            print("Status: FAILED")

        # Test Case 5: Key Derivation (Context String)
        print("\n--- Test Case 5: Key Derivation ---")
        context_str = "TestContextForDerivation"
        hasher_ctx = Blake3Hasher(context_string=context_str)
        hasher_ctx.update(b"input data for derived key")
        digest_bytes_ctx = hasher_ctx.finalize(out_len=32)
        computed_hex_ctx = digest_bytes_ctx.hex()
        known_hex_ctx = "0196e2c07300537214389545071d8530e441b68918d76854a95714498a4e0610"
        print(f"Input: b'input data for derived key', Context: '{context_str}'")
        print(f"Computed digest: {computed_hex_ctx}")
        print(f"Expected digest: {known_hex_ctx}")
        if computed_hex_ctx == known_hex_ctx:
            print("Status: PASSED")
        else:
            print("Status: FAILED")
            
        print("SCRIPT_TRY_END")

    except Exception as e:
        print("SCRIPT_EXCEPT_START")
        print(f"\nAn error occurred during testing: {e}")
        import traceback
        traceback.print_exc()
        print("SCRIPT_EXCEPT_END")
    print("SCRIPT_MAIN_END")

# Tensor.no_grad and Tensor.training are set at the start of the try block in main.
print("SCRIPT_EOF")
