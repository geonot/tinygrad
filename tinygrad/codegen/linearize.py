def finalize(sink:UOp) -> UOp:
  if sink.op is not Ops.BLOCK or not all(x.op in DONT_PLACE_IN_BLOCK for x in sink.src):
    raise RuntimeError("linearize failure")

  # place the early things
  lst = sorted(dedup(sink.src), key=lambda x: x.tuplize) + list(sink.arg.lst)

  if __debug__: type_verify(lst)

  # CONTROL FLAGS (e.g., from getenv or a config object)
  # Use a helper like getenv from tinygrad.helpers for cleaner access
  from tinygrad.helpers import getenv

  ENABLE_TILING = getenv("ENABLE_TILING", 1)
  ENABLE_VECTORIZATION = getenv("ENABLE_VECTORIZATION", 1)

  if ENABLE_TILING:
    print("Tiling pass enabled.")
    MAX_TILING_ITERATIONS = 10 # Safeguard against infinite loops
    tiling_iterations = 0
    processed_tilable_loops = set()

    while tiling_iterations < MAX_TILING_ITERATIONS:
      modified_in_pass = False
      found_candidate_in_pass = False # To break if no active candidates are found
      for i, uop in enumerate(lst):
        if uop.op == Ops.RANGE and uop not in processed_tilable_loops:
          found_candidate_in_pass = True
          has_wmma = False # Current heuristic: only tile loops with WMMA
          end_range_idx = -1
          for j_inner in range(i + 1, len(lst)):
            if lst[j_inner].op == Ops.ENDRANGE and lst[j_inner].src[0] == uop:
              end_range_idx = j_inner
              break

          if end_range_idx != -1:
            for k_inner in range(i + 1, end_range_idx):
              if lst[k_inner].op == Ops.WMMA:
                has_wmma = True
                break

          if has_wmma:
            print(f"Tiling candidate: Loop {uop.arg} (idx {i}) contains WMMA.")
            tile_val = 16 # Fixed tile size for now

            original_uop_ref = uop

            lst, acc_replace_map = tile_loop(lst, i, tile_val)

            if acc_replace_map:
              print(f"Applying accumulator replacements post-tiling: {[(k.arg if hasattr(k,'arg') else id(k), v.arg if hasattr(v,'arg') else id(v)) for k,v in acc_replace_map.items()]}")
              new_lst_after_acc_sub = []
              for current_uop_in_lst in lst: # Iterate over the list returned by tile_loop
                  substituted_uop = current_uop_in_lst.substitute(acc_replace_map)
                  new_lst_after_acc_sub.append(substituted_uop)
              lst = new_lst_after_acc_sub

            print(f"Loop tiling applied for loop {original_uop_ref.arg}. New lst length: {len(lst)}.")
            processed_tilable_loops.add(original_uop_ref)
            modified_in_pass = True
            break

      tiling_iterations += 1
      if not modified_in_pass:
        if not found_candidate_in_pass and tiling_iterations == 1 : # Only print if no candidates at all on first pass
             print(f"No tiling candidates found in the UOp list.")
        elif found_candidate_in_pass : # Candidates were there, but none were tiled (e.g. already processed)
             print(f"Completed a tiling pass with no modifications made. Iteration: {tiling_iterations}")
        break
      if tiling_iterations == MAX_TILING_ITERATIONS:
        print("Warning: Reached max tiling iterations.")
        break
  else:
    print("Tiling pass disabled.")

  if ENABLE_VECTORIZATION:
    VEC_FACTOR = getenv("VEC_FACTOR", 4)
    if VEC_FACTOR > 1:
        print(f"Vectorization pass enabled with factor {VEC_FACTOR}.")
        lst = vectorize_pass(lst, VEC_FACTOR)
    else:
        print("Vectorization pass disabled (VEC_FACTOR <= 1).")
  else:
    print("Vectorization pass disabled.")

  return UOp(Ops.BLOCKFINAL, arg=BasicBlock2(tuple(lst)))

pm_finalize = PatternMatcher([(UPat(Ops.BLOCK, name="sink"), finalize)])

# --- Vectorization Pass ---
def vectorize_pass(uops: list[UOp], vector_factor: int) -> list[UOp]:
    print(f"Starting vectorization pass with factor {vector_factor}.")

    MAX_VECTORIZATION_ITERATIONS = 10 # Safeguard for the outer pass
    pass_iteration = 0
    overall_modified = True

    new_uops_list = list(uops)

    while overall_modified and pass_iteration < MAX_VECTORIZATION_ITERATIONS:
        overall_modified = False
        pass_iteration += 1
        print(f" Vectorization pass iteration: {pass_iteration}")

        # Find all loop ranges first to avoid issues with list modification during iteration
        loop_ranges_info = [] # Store (range_uop, start_idx, end_idx)
        for i, uop in enumerate(new_uops_list):
            if uop.op == Ops.RANGE:
                # Check if it's an innermost loop (no other RANGE ops inside)
                is_innermost = True
                end_idx = -1
                for j in range(i + 1, len(new_uops_list)):
                    if new_uops_list[j].op == Ops.RANGE: # Found a nested RANGE
                        is_innermost = False
                        break
                    if new_uops_list[j].op == Ops.ENDRANGE and new_uops_list[j].src[0] == uop:
                        end_idx = j
                        break
                if end_idx == -1: # Should not happen for well-formed UOps
                    print(f"Warning: Could not find ENDRANGE for RANGE {uop.arg}")
                    continue

                if is_innermost:
                    loop_ranges_info.append({"range_uop": uop, "start_idx": i, "end_idx": end_idx})

        if not loop_ranges_info:
            print(" No innermost loops found to vectorize.")
            break

        for loop_info in loop_ranges_info:
            range_uop = loop_info["range_uop"]
            start_idx = loop_info["start_idx"]
            end_idx = loop_info["end_idx"] # end_idx of ENDRANGE

            # Extract loop body (uops between RANGE and ENDRANGE)
            # Indices are relative to new_uops_list at the time of loop_info collection.
            # If list is modified, these indices might become stale for subsequent loops in the same pass.
            # This is why we break and restart if a modification occurs.

            # Check if start_idx and end_idx are still valid for new_uops_list length
            if not (start_idx < len(new_uops_list) and end_idx < len(new_uops_list) and \
                    new_uops_list[start_idx] == range_uop and \
                    new_uops_list[end_idx].op == Ops.ENDRANGE and new_uops_list[end_idx].src[0] == range_uop):
                print(f"  Loop {range_uop.arg} indices became stale, will re-evaluate in next pass if needed.")
                # This loop's info is stale due to a previous vectorization in the same pass.
                # Set overall_modified to force another pass to re-evaluate loops.
                overall_modified = True
                break # Break from processing loop_ranges_info, restart while overall_modified

            loop_body = new_uops_list[start_idx+1:end_idx]

            print(f"  Attempting to vectorize loop body for RANGE {range_uop.arg} (indices {start_idx+1}-{end_idx-1})")

            new_loop_body, body_modified = vectorize_loop_body(loop_body, range_uop, vector_factor, new_uops_list)

            if body_modified:
                print(f"  Loop body for RANGE {range_uop.arg} was vectorized. Reconstructing UOp list.")
                # Replace the old loop body with the new one
                new_uops_list = new_uops_list[:start_idx+1] + new_loop_body + new_uops_list[end_idx:]
                overall_modified = True
                # Break from processing loop_ranges_info, restart the while overall_modified loop
                # to re-scan everything from the modified list.
                break

        if overall_modified: # If a modification happened in the inner loop processing
            continue # Restart the while loop to re-evaluate all loops

    if pass_iteration == MAX_VECTORIZATION_ITERATIONS:
        print("Warning: Vectorization pass reached max iterations.")

    print("Vectorization pass finished.")
    return new_uops_list

def vectorize_loop_body(loop_body: list[UOp], range_uop: UOp, vector_factor: int, full_uops_list: list[UOp]) -> tuple[list[UOp], bool]:
    """
    Attempts to vectorize operations within a loop body.
    full_uops_list is needed for context like get_max_loop_idx or creating new unique UOps.
    Returns: new_loop_body, modified_flag
    """
    print(f" vectorize_loop_body called for loop {range_uop.arg} (placeholder).")
    modified = False

    # TODO: Implement actual vectorization logic:
    # 1. Detect N contiguous scalar loads -> 1 vector load + N GEPs
    # 2. Detect N scalar ops (ALU) using results of vectorized loads -> 1 vector ALU + N GEPs
    # 3. Detect N scalar values to be stored contiguously -> N-1 VECTORIZE + 1 vector store

    new_loop_body = list(loop_body) # Work on a copy
    i = 0
    while i < len(new_loop_body) - (vector_factor - 1): # Need at least vector_factor elements
        uop0 = new_loop_body[i]
        if uop0.op is Ops.LOAD:
            # Try to find vector_factor contiguous loads
            scalar_loads_to_vectorize = [uop0]
            can_vectorize_this_sequence = True

            if not (uop0.dtype.count == 1): # Must be scalar loads
                i += 1
                continue

            scalar_dtype = uop0.dtype

            # Check next (vector_factor - 1) loads
            for k in range(1, vector_factor):
                if i + k >= len(new_loop_body):
                    can_vectorize_this_sequence = False
                    break

                uop_k = new_loop_body[i+k]
                if not (uop_k.op is Ops.LOAD and uop_k.dtype == scalar_dtype):
                    can_vectorize_this_sequence = False
                    break

                # Basic contiguity check (highly simplified placeholder)
                # Assumes idx_op.src[0] is buffer, idx_op.src[1] is index value UOp
                idx0_op = scalar_loads_to_vectorize[0].src[0] # Ops.INDEX for load0
                idxk_op = uop_k.src[0] # Ops.INDEX for loadk

                if not (idx0_op.op is Ops.INDEX and idxk_op.op is Ops.INDEX and \
                        idx0_op.src[0] == idxk_op.src[0] and \
                        idx0_op.arg == idxk_op.arg): # Same buffer and same Ops.INDEX arg (e.g. stride/etc. if used)
                    can_vectorize_this_sequence = False
                    break

                # Improved Contiguity Check:
                # Check if index_val_k is effectively index_val_0 + CONST(k)
                index_val_0_uop = idx0_op.src[1]
                index_val_k_uop = idxk_op.src[1]

                # Expected: index_val_k_uop should be ADD(index_val_0_uop, CONST(k)) or ADD(CONST(k), index_val_0_uop)
                is_contiguous_k = False
                if index_val_k_uop.op is Ops.ADD and len(index_val_k_uop.src) == 2:
                    const_k_uop = UOp.const(index_val_0_uop.dtype, k) # Assuming index dtype matches

                    # Check form ADD(index_val_0_uop, CONST(k))
                    if index_val_k_uop.src[0] == index_val_0_uop and \
                       index_val_k_uop.src[1].op is Ops.CONST and \
                       index_val_k_uop.src[1].arg == k:
                        is_contiguous_k = True
                    # Check form ADD(CONST(k), index_val_0_uop)
                    elif index_val_k_uop.src[1] == index_val_0_uop and \
                         index_val_k_uop.src[0].op is Ops.CONST and \
                         index_val_k_uop.src[0].arg == k:
                        is_contiguous_k = True

                if not is_contiguous_k:
                    # Print for debugging what was found instead
                    # print(f"    Load {k} not contiguous: {idxk_op.src[1].op} {idxk_op.src[1].arg if hasattr(idxk_op.src[1],'arg') else ''} srcs={idxk_op.src[1].src if hasattr(idxk_op.src[1],'src') else '' }")
                    can_vectorize_this_sequence = False
                    break

                scalar_loads_to_vectorize.append(uop_k)

            if can_vectorize_this_sequence and len(scalar_loads_to_vectorize) == vector_factor:
                print(f"Found candidate sequence of {vector_factor} loads starting at uop {i} for loop {range_uop.arg}")

                # Perform vectorization
                load0_op = scalar_loads_to_vectorize[0]
                idx0_op = load0_op.src[0] # This is an Ops.INDEX
                buffer_uop = idx0_op.src[0] # Original buffer (e.g. DEFINE_GLOBAL)
                index_val0_uop = idx0_op.src[1] # UOp calculating the index for the first load

                vector_dtype = scalar_dtype.vec(vector_factor)

                # 1. Bitcast buffer pointer to vector pointer type
                # PtrDType now takes a DType directly.
                ptr_to_vector_dtype = PtrDType(vector_dtype)
                bitcast_buffer_uop = UOp(Ops.BITCAST, ptr_to_vector_dtype, (buffer_uop,))

                # 2. Create new INDEX UOp for the vector load (points to start of vector)
                # The index value (src[1]) should be the same as the first scalar load's index value.
                # The arg of original INDEX might contain stride/multiplier, preserve it.
                vector_idx_uop = UOp(Ops.INDEX, ptr_to_vector_dtype, (bitcast_buffer_uop, index_val0_uop), arg=idx0_op.arg)

                # 3. Create the vector LOAD UOp
                vector_load_uop = UOp(Ops.LOAD, vector_dtype, (vector_idx_uop,))

                # 4. Create GEP UOps to extract scalar values
                gep_uops = []
                for k_gep in range(vector_factor):
                    gep_uops.append(UOp(Ops.GEP, scalar_dtype, (vector_load_uop,), arg=(k_gep,)))

                # 5. Build substitution map for original scalar loads
                substitution_map = {}
                for k_map in range(vector_factor):
                    substitution_map[scalar_loads_to_vectorize[k_map]] = gep_uops[k_map]

                print(f"  Vectorization: {vector_load_uop.dtype} <- {buffer_uop.arg if hasattr(buffer_uop,'arg') else 'VAR'}[{index_val0_uop.arg if hasattr(index_val0_uop,'arg') else 'VAR_IDX'}]")
                for old, new in substitution_map.items(): print(f"    Replacing {old.arg if old.arg else id(old)} with GEP from {new.src[0].arg if new.src[0].arg else id(new.src[0])}")

                # 6. Reconstruct the UOp list for this part of the loop body
                # Insert new UOps: bitcast, vector_idx, vector_load, GEPs
                # Remove old scalar loads
                # Substitute uses of old loads in subsequent uops

                # This part needs to be done carefully.
                # We replace the 'i'th UOp with the new sequence, and remove the next N-1 UOps.
                # Then substitute in the rest.

                transformed_uops = [bitcast_buffer_uop, vector_idx_uop, vector_load_uop] + gep_uops

                # The uops from new_loop_body[i+vector_factor:] need to be substituted
                rest_of_body = new_loop_body[i+vector_factor:]
                substituted_rest_of_body = [u.substitute(substitution_map) for u in rest_of_body]

                new_loop_body = new_loop_body[:i] + transformed_uops + substituted_rest_of_body

                modified = True
                # Restart scan for this loop body from the beginning
                # because dependencies might have changed allowing further vectorization.
                # The current 'i' is invalidated.
                i = 0
                continue # Restart while loop for new_loop_body

        # Attempt to vectorize STORES
        uop0_store = new_loop_body[i]
        if uop0_store.op is Ops.STORE:
            scalar_stores_to_vectorize = [uop0_store]
            can_vectorize_store_sequence = True

            if not (uop0_store.src[1].dtype.count == 1): # Value being stored must be scalar
                i += 1
                continue

            scalar_dtype_store = uop0_store.src[1].dtype # Dtype of the value being stored

            for k in range(1, vector_factor):
                if i + k >= len(new_loop_body):
                    can_vectorize_store_sequence = False
                    break
                uop_k_store = new_loop_body[i+k]

                if not (uop_k_store.op is Ops.STORE and uop_k_store.src[1].dtype == scalar_dtype_store):
                    can_vectorize_store_sequence = False
                    break

                idx0_op_store = scalar_stores_to_vectorize[0].src[0] # Ops.INDEX for store0
                idxk_op_store = uop_k_store.src[0]                   # Ops.INDEX for storek

                if not (idx0_op_store.op is Ops.INDEX and idxk_op_store.op is Ops.INDEX and \
                        idx0_op_store.src[0] == idxk_op_store.src[0] and \
                        idx0_op_store.arg == idxk_op_store.arg): # Same buffer and INDEX args
                    can_vectorize_store_sequence = False
                    break

                index_val_0_uop_store = idx0_op_store.src[1]
                index_val_k_uop_store = idxk_op_store.src[1]
                is_contiguous_k_store = False
                if index_val_k_uop_store.op is Ops.ADD and len(index_val_k_uop_store.src) == 2:
                    if (index_val_k_uop_store.src[0] == index_val_0_uop_store and \
                        index_val_k_uop_store.src[1].op is Ops.CONST and index_val_k_uop_store.src[1].arg == k) or \
                       (index_val_k_uop_store.src[1] == index_val_0_uop_store and \
                        index_val_k_uop_store.src[0].op is Ops.CONST and index_val_k_uop_store.src[0].arg == k) :
                        is_contiguous_k_store = True

                if not is_contiguous_k_store:
                    can_vectorize_store_sequence = False
                    break
                scalar_stores_to_vectorize.append(uop_k_store)

            if can_vectorize_store_sequence and len(scalar_stores_to_vectorize) == vector_factor:
                print(f"Found candidate sequence of {vector_factor} stores starting at uop {i} for loop {range_uop.arg}")

                store0_op = scalar_stores_to_vectorize[0]
                idx0_op_store = store0_op.src[0]      # Ops.INDEX for first store
                buffer_uop_store = idx0_op_store.src[0] # Original buffer
                index_val0_uop_store = idx0_op_store.src[1] # Index UOp for first store

                vector_dtype_store = scalar_dtype_store.vec(vector_factor)
                ptr_to_vector_dtype_store = PtrDType(vector_dtype_store)

                # 1. Collect scalar values to be stored
                scalar_value_uops = [s.src[1] for s in scalar_stores_to_vectorize]

                # 2. Create Ops.VECTORIZE to form the vector value
                vector_value_uop = UOp(Ops.VECTORIZE, vector_dtype_store, tuple(scalar_value_uops))

                # 3. Bitcast buffer pointer
                bitcast_buffer_uop_store = UOp(Ops.BITCAST, ptr_to_vector_dtype_store, (buffer_uop_store,))

                # 4. Create new INDEX UOp for the vector store
                vector_idx_uop_store = UOp(Ops.INDEX, ptr_to_vector_dtype_store, (bitcast_buffer_uop_store, index_val0_uop_store), arg=idx0_op_store.arg)

                # 5. Create the vector STORE UOp
                vector_store_uop = UOp(Ops.STORE, dtypes.void, (vector_idx_uop_store, vector_value_uop))

                print(f"  Vectorization: STORE {vector_store_uop.src[1].dtype} -> {buffer_uop_store.arg if hasattr(buffer_uop_store,'arg') else 'VAR'}[{index_val0_uop_store.arg if hasattr(index_val0_uop_store,'arg') else 'VAR_IDX'}]")

                transformed_uops_store = [vector_value_uop, bitcast_buffer_uop_store, vector_idx_uop_store, vector_store_uop]

                # Stores don't produce values that need substitution in later ops in the same way loads do.
                # We just replace the sequence of store UOps.
                rest_of_body_store = new_loop_body[i+vector_factor:]
                # No substitution needed for store results, but if any of the *scalar_value_uops* were
                # from a substitution map (e.g. GEPs from a vector load), that's handled before this.

                new_loop_body = new_loop_body[:i] + transformed_uops_store + rest_of_body_store
                modified = True
                i = 0 # Restart scan
                continue

        # Attempt to vectorize ALU operations
        uop0_alu = new_loop_body[i]
        if uop0_alu.op in GroupOp.ALU and uop0_alu.dtype.count == 1: # Process scalar ALU ops
            scalar_alu_ops_to_vectorize = [uop0_alu]
            can_vectorize_alu_sequence = True

            # Check next (vector_factor - 1) ALU ops
            for k in range(1, vector_factor):
                if i + k >= len(new_loop_body):
                    can_vectorize_alu_sequence = False
                    break
                uop_k_alu = new_loop_body[i+k]

                # Check if it's the same ALU operation and same scalar dtype
                if not (uop_k_alu.op == uop0_alu.op and uop_k_alu.dtype == uop0_alu.dtype):
                    can_vectorize_alu_sequence = False
                    break

                # Check if inputs are from corresponding GEPs of same vector source(s)
                # Example: add0 = ADD(gep(vL,0), gep(vR,0)), add1 = ADD(gep(vL,1), gep(vR,1))
                # This requires checking uop0_alu.src and uop_k_alu.src
                if len(uop0_alu.src) != len(uop_k_alu.src)): # Should have same number of sources
                    can_vectorize_alu_sequence = False
                    break

                # This is a placeholder for the GEP source check. Highly simplified.
                # A full check would trace back sources to common vector UOps and matching GEP indices.
                # For now, just check if the ops and types match, rely on prior load vectorization.
                # This will incorrectly vectorize if ALU inputs are not from corresponding vector lanes.
                # MAJOR SIMPLIFICATION HERE.

                # Heuristic: if all sources of uop0_alu are GEPs, and all sources of uop_k_alu are GEPs,
                # and for each pair of sources (src_j_uop0, src_j_uopk), they are GEPs from the
                # same vector_load_uop, and their GEP index is 0 for src_j_uop0 and k for src_j_uopk.

                # For now, we'll assume if we have N identical scalar ALUs, their inputs *might*
                # have been set up by previous vector load GEPs.

                scalar_alu_ops_to_vectorize.append(uop_k_alu)

            if can_vectorize_alu_sequence and len(scalar_alu_ops_to_vectorize) == vector_factor:
                print(f"Found candidate sequence of {vector_factor} ALU ops ({uop0_alu.op}) starting at uop {i} for loop {range_uop.arg}")

                # Assuming sources are already vectorized or GEPs from vectorized sources
                # This part needs to identify the actual vector sources for the new vector ALU op.
                # Example: if src0 of uop0_alu is GEP(vL0, 0) and src0 of uop1_alu is GEP(vL0, 1),
                # then the new vector ALU op's src0 should be vL0.

                # Placeholder for identifying true vector sources. This is complex.
                # For now, create a dummy vector ALU op if the pattern matches.
                # This will not be correct without proper source tracing.

                # Let's try to make it slightly more robust:
                # Check if all first sources (src[0]) of the scalar ALUs are GEPs from the same vector,
                # with sequential GEP indices. Same for second sources (src[1]) if they exist.

                vector_sources = []
                possible_alu_vectorization = True
                for src_idx in range(len(uop0_alu.src)): # For each source of the ALU op (e.g., 2 for binary)
                    first_gep_src_uop = scalar_alu_ops_to_vectorize[0].src[src_idx]
                    if not (first_gep_src_uop.op is Ops.GEP and first_gep_src_uop.arg == (0,)):
                        possible_alu_vectorization = False; break

                    vector_source_candidate = first_gep_src_uop.src[0] # The vector UOp feeding the GEPs

                    for k_alu in range(1, vector_factor):
                        current_alu_src_uop = scalar_alu_ops_to_vectorize[k_alu].src[src_idx]
                        if not (current_alu_src_uop.op is Ops.GEP and \
                                current_alu_src_uop.src[0] == vector_source_candidate and \
                                current_alu_src_uop.arg == (k_alu,)):
                            possible_alu_vectorization = False; break
                    if not possible_alu_vectorization: # If it's not a GEP sequence for this src_idx
                        # Check if it's a loop-invariant scalar (same UOp for all scalar ALUs in sequence)
                        # or a const that can be broadcast.
                        first_src_for_alu_operand = scalar_alu_ops_to_vectorize[0].src[src_idx]
                        is_broadcastable_scalar = True
                        if first_src_for_alu_operand.op is Ops.GEP: # Cannot be GEP if it failed GEP sequence check
                            is_broadcastable_scalar = False

                        if is_broadcastable_scalar: # Check if it's truly loop invariant for this sequence
                            for k_alu_check in range(1, vector_factor):
                                if scalar_alu_ops_to_vectorize[k_alu_check].src[src_idx] != first_src_for_alu_operand:
                                    is_broadcastable_scalar = False; break

                        if is_broadcastable_scalar:
                            print(f"    ALU src[{src_idx}] is a broadcastable scalar: {first_src_for_alu_operand.op} {first_src_for_alu_operand.arg if hasattr(first_src_for_alu_operand,'arg') else ''}")
                            # Create a VECTORIZE UOp to broadcast this scalar
                            # The VECTORIZE UOp for broadcast takes scalar as src and its dtype is vector
                            scalar_src_dtype = first_src_for_alu_operand.dtype
                            broadcast_vector_dtype = scalar_src_dtype.vec(vector_factor)
                            # Ops.VECTORIZE src must be a tuple. For broadcast, it's (scalar_uop,)
                            broadcast_uop = UOp(Ops.VECTORIZE, broadcast_vector_dtype, (first_src_for_alu_operand,))
                            vector_sources.append(broadcast_uop)
                            # Add this broadcast_uop to a list to be inserted before the vector_alu_uop
                            # This will be handled if transformed_uops_alu includes it.
                        else: # Not a GEP sequence and not a broadcastable scalar
                            possible_alu_vectorization = False; break
                    else: # It was a GEP sequence for this src_idx
                        vector_sources.append(vector_source_candidate)

                if possible_alu_vectorization:
                    scalar_dtype_alu = uop0_alu.dtype
                    vector_dtype_alu = scalar_dtype_alu.vec(vector_factor)

                    # Create the vector ALU UOp
                    vector_alu_uop = UOp(uop0_alu.op, vector_dtype_alu, tuple(vector_sources), arg=uop0_alu.arg)

                    # Create GEP UOps for its results
                    gep_alu_uops = [UOp(Ops.GEP, scalar_dtype_alu, (vector_alu_uop,), arg=(k_gep,)) for k_gep in range(vector_factor)]

                    substitution_map_alu = {scalar_alu_ops_to_vectorize[k_map]: gep_alu_uops[k_map] for k_map in range(vector_factor)}

                    print(f"  Vectorization: ALU {vector_alu_uop.op} {vector_alu_uop.dtype}")

                    # Collect all newly created UOps for this ALU vectorization
                    # These are: any broadcast UOps created, the vector ALU UOp, and the GEPs for results.
                    transformed_uops_alu = []
                    for v_src in vector_sources:
                        # Check if v_src is a broadcast UOp that was newly created in this step
                        # A simple check: it's an Ops.VECTORIZE and its source is not a GEP
                        # (because GEPs would mean it came from a vectorized load, not a broadcast scalar)
                        if v_src.op is Ops.VECTORIZE:
                             # Check if its input was one of the original scalar ALU inputs that was broadcasted
                             is_a_newly_created_broadcast = False
                             for orig_alu_src_uop in uop0_alu.src: # Check sources of the first scalar ALU
                                 if v_src.src[0] == orig_alu_src_uop and orig_alu_src_uop.op is not Ops.GEP:
                                     is_a_newly_created_broadcast = True
                                     break
                             if is_a_newly_created_broadcast:
                                transformed_uops_alu.append(v_src)

                    transformed_uops_alu.append(vector_alu_uop)
                    transformed_uops_alu.extend(gep_alu_uops)

                    rest_of_body_alu = new_loop_body[i+vector_factor:]
                    substituted_rest_of_body_alu = [u.substitute(substitution_map_alu) for u in rest_of_body_alu]

                    new_loop_body = new_loop_body[:i] + transformed_uops_alu + substituted_rest_of_body_alu
                    modified = True
                    i = 0 # Restart scan
                    continue
                else:
                    print(f"  ALU sequence starting at {i} not vectorizable due to GEP source mismatch.")


        i += 1 # Move to next UOp if no vectorization happened at current 'i'

    return new_loop_body, modified

def get_max_loop_idx(uops: list[UOp]) -> int:
  """Scans UOps to find the maximum index used in RANGE arg or SPECIAL ('loopvar', idx, ...) arg."""
  max_idx = -1
  for uop in uops:
    if uop.op == Ops.RANGE:
      if isinstance(uop.arg, int) and uop.arg > max_idx:
        max_idx = uop.arg
    elif uop.op == Ops.SPECIAL and isinstance(uop.arg, tuple) and len(uop.arg) > 1 and uop.arg[0] == "loopvar":
      if isinstance(uop.arg[1], int) and uop.arg[1] > max_idx:
        max_idx = uop.arg[1]
  return max_idx

def tile_loop(uops: list[UOp], loop_start_idx: int, tile_val: int) -> tuple[list[UOp], dict[UOp, UOp]]:
  """
  Tiles a single loop within a list of UOps into an outer and inner loop.
  Returns the new list of UOps and a map of original DEFINE_ACC UOps to their new global counterparts.
  """
  original_range_uop = uops[loop_start_idx]
  loop_end_idx = -1
  for i in range(loop_start_idx + 1, len(uops)):
    if uops[i].op == Ops.ENDRANGE and uops[i].src[0] == original_range_uop:
      loop_end_idx = i
      break
  if loop_end_idx == -1:
    raise ValueError(f"CANNOT FIND ENDRANGE for RANGE UOp: {original_range_uop}")

  original_endrange_uop = uops[loop_end_idx]
  loop_body = uops[loop_start_idx+1:loop_end_idx]

  original_limit_uop = original_range_uop.src[0]
  original_idx = original_range_uop.arg
  original_dtype = original_range_uop.dtype # Assuming loop var dtype is same as range dtype

  const_tile_uop = UOp.const(original_dtype, tile_val)
  const_one_uop = UOp.const(original_dtype, 1)
  sum_uop = UOp(Ops.ADD, original_dtype, (original_limit_uop, const_tile_uop))
  numerator_uop = UOp(Ops.SUB, original_dtype, (sum_uop, const_one_uop))
  outer_limit_uop = UOp(Ops.IDIV, original_dtype, (numerator_uop, const_tile_uop))

  max_existing_idx = get_max_loop_idx(uops)
  outer_loop_idx = max_existing_idx + 1
  inner_loop_idx = max_existing_idx + 2
  print(f"Original idx: {original_idx}, Max existing: {max_existing_idx}, Outer new: {outer_loop_idx}, Inner new: {inner_loop_idx}")

  outer_range_uop = UOp(Ops.RANGE, original_dtype, src=(outer_limit_uop,), arg=outer_loop_idx)
  outer_endrange_uop = UOp(Ops.ENDRANGE, src=(outer_range_uop,))

  inner_limit_uop = UOp.const(original_dtype, tile_val)
  inner_range_uop = UOp(Ops.RANGE, original_dtype, src=(inner_limit_uop,), arg=inner_loop_idx)
  inner_endrange_uop = UOp(Ops.ENDRANGE, src=(inner_range_uop,))

  acc_replace_map: dict[UOp, UOp] = {}
  global_acc_defines: list[UOp] = []
  tile_local_acc_inits: list[UOp] = []
  acc_post_inner_loop_ops: list[UOp] = []

  for uop_in_body in loop_body:
    if uop_in_body.op is Ops.ASSIGN and uop_in_body.src[0].op is Ops.DEFINE_ACC:
      original_acc_uop = uop_in_body.src[0]
      # Simple check: if this DEFINE_ACC is involved in an ASSIGN in this loop, transform it.
      # More robust: check if original_range_uop is in original_acc_uop.src[2:] (loops it depends on)
      print(f"Processing DEFINE_ACC: {original_acc_uop.arg if original_acc_uop.arg else id(original_acc_uop)} for tiling.")

      acc_dtype = original_acc_uop.dtype
      acc_reduce_op_enum = original_acc_uop.arg # e.g. Ops.SUM
      acc_initial_val_uop = original_acc_uop.src[0]

      # Create global accumulator
      # Its src should include the outer_range_uop for PHI node generation.
      global_acc_uop = UOp(Ops.DEFINE_ACC, acc_dtype, src=(acc_initial_val_uop, outer_range_uop), arg=acc_reduce_op_enum)
      global_acc_defines.append(global_acc_uop)
      acc_replace_map[original_acc_uop] = global_acc_uop # Map old to new global for external users
      print(f"  Created global acc: {global_acc_uop.arg if global_acc_uop.arg else id(global_acc_uop)} linked to loop {outer_range_uop.arg}")

      # The original_acc_uop becomes tile-local.
      # It needs to be linked to the inner_range_uop for its own PHI node.
      # And it needs re-initialization inside the outer loop.
      # We replace original_acc_uop in the loop_body processing with a new tile_local_acc definition
      # that is correctly scoped.

      # Create new tile-local DEFINE_ACC UOp linked to inner_range_uop
      # This new UOp will be used in the modified_loop_body.
      tile_local_acc_definition = UOp(Ops.DEFINE_ACC, acc_dtype, src=(acc_initial_val_uop, inner_range_uop), arg=acc_reduce_op_enum)
      # This tile_local_acc_definition is what the ASSIGNs in the inner loop body should target.
      # So, assignments to original_acc_uop inside loop_body need to be changed to target tile_local_acc_definition.
      # This is done by adding original_acc_uop -> tile_local_acc_definition to the substitution_map for _modify_loop_body_for_tiling.

      # Create tile-local accumulator initialization (assign initial value to the new tile_local_acc_definition)
      init_tile_local_assign = UOp(Ops.ASSIGN, acc_dtype, src=(tile_local_acc_definition, acc_initial_val_uop))
      tile_local_acc_inits.append(tile_local_acc_definition) # Add definition first
      tile_local_acc_inits.append(init_tile_local_assign)
      print(f"  Created tile-local acc definition: {tile_local_acc_definition.arg if tile_local_acc_definition.arg else id(tile_local_acc_definition)} linked to loop {inner_range_uop.arg}")
      print(f"  Created tile-local init assign for it.")

      # Create ops to accumulate tile-local result into global accumulator
      # The value used is from tile_local_acc_definition.
      reduce_op = UOp(acc_reduce_op_enum, acc_dtype, src=(global_acc_uop, tile_local_acc_definition))
      assign_to_global_op = UOp(Ops.ASSIGN, acc_dtype, src=(global_acc_uop, reduce_op))
      acc_post_inner_loop_ops.extend([reduce_op, assign_to_global_op])
      print(f"  Created ops to accumulate tile result to global_acc")

  modified_loop_body, new_uops_for_index_calc = _modify_loop_body_for_tiling(
                                                    loop_body,
                                                    loop_body,
                                                    original_range_uop,
                                                    outer_range_uop,
                                                    inner_range_uop,
                                                    tile_val,
                                                    uops[:loop_start_idx],
                                                    # Pass map to update original_acc assignments to tile_local_acc
                                                    {original_acc_uop: tile_local_acc_definition if 'tile_local_acc_definition' in locals() else original_acc_uop for original_acc_uop in acc_replace_map.keys()}
                                                    )

  new_uops = uops[:loop_start_idx] + \
             global_acc_defines + \
             [outer_range_uop] + \
             tile_local_acc_inits + \
             [inner_range_uop] + \
             new_uops_for_index_calc + modified_loop_body + \
             [inner_endrange_uop] + \
             acc_post_inner_loop_ops + \
             [outer_endrange_uop] + \
             uops[loop_end_idx+1:]

  print(f"Tiling loop: {original_range_uop} with tile_val {tile_val}")
  print("Loop tiling structure created (1D). Index modification attempted. Accumulator handling added.")
  # Debug printing (adjust as needed)
  # ... (debug print logic from before, ensure indices are valid for new_uops)

  return new_uops, acc_replace_map

def _get_loop_var_uop(loop_idx: int, dtype: DType, tag:str="var") -> UOp:
  assert isinstance(loop_idx, int), f"loop_idx must be an int, got {loop_idx}"
  return UOp(Ops.SPECIAL, dtype, src=(), arg=("loopvar", loop_idx, tag))

def _create_tiled_index_uop(original_loop_idx: int,
                            outer_loop_var_uop: UOp,
                            inner_loop_var_uop: UOp,
                            tile_val: int,
                            dtype: DType) -> tuple[UOp, list[UOp]]:
  created_uops = []
  const_tile_uop = UOp.const(dtype, tile_val)
  term1 = UOp(Ops.MUL, dtype, (outer_loop_var_uop, const_tile_uop))
  created_uops.append(term1)
  new_index_val_uop = UOp(Ops.ADD, dtype, (term1, inner_loop_var_uop))
  created_uops.append(new_index_val_uop)
  # print(f"Created new_index_val_uop: {new_index_val_uop.op} arg={new_index_val_uop.arg} dtype={new_index_val_uop.dtype} " +
  #       f"src={[(s.op, s.arg if s.op==Ops.CONST else s.arg, s.dtype) for s in new_index_val_uop.src]}")
  return new_index_val_uop, created_uops

def _modify_loop_body_for_tiling(loop_body: list[UOp],
                                 original_range_uop: UOp,
                                 outer_range_uop: UOp,
                                 inner_range_uop: UOp,
                                 tile_val: int,
                                 preceding_uops: list[UOp],
                                 acc_target_remap: dict[UOp, UOp]) -> tuple[list[UOp], list[UOp]]:
  new_loop_body = []
  inserted_calc_uops = []
  original_idx_int = original_range_uop.arg
  dtype = original_range_uop.dtype
  outer_loop_var_val_uop = _get_loop_var_uop(outer_range_uop.arg, dtype, tag="out_var")
  inner_loop_var_val_uop = _get_loop_var_uop(inner_range_uop.arg, dtype, tag="in_var")
  calculated_new_idx_uop, calc_uops = _create_tiled_index_uop(
      original_idx_int,
      outer_loop_var_val_uop,
      inner_loop_var_val_uop,
      tile_val,
      dtype
  )
  inserted_calc_uops.extend(calc_uops)
  original_loop_var_uop = _get_loop_var_uop(original_idx_int, dtype, tag="orig_var_to_replace")
  # This substitution map is for the loop induction variable.
  idx_substitution_map = {original_loop_var_uop: calculated_new_idx_uop}

  # print(f"Idx Substitution map: {original_loop_var_uop.arg} -> {calculated_new_idx_uop.op} {calculated_new_idx_uop.arg}")
  # if acc_target_remap:
  #   print(f"Acc Target Remap: {[(k.arg, v.arg) for k,v in acc_target_remap.items()]}")

  for uop_in_body in loop_body:
    # First, handle re-targeting of ASSIGN operations for accumulators
    current_uop_to_process = uop_in_body
    if current_uop_to_process.op is Ops.ASSIGN and current_uop_to_process.src[0] in acc_target_remap:
      # This ASSIGN was to an original accumulator. Re-target it to the new tile-local accumulator.
      new_target_acc = acc_target_remap[current_uop_to_process.src[0]]
      # value_being_assigned (uop_in_body.src[1]) will be processed by substitute below for index vars
      current_uop_to_process = UOp(Ops.ASSIGN, current_uop_to_process.dtype, (new_target_acc, current_uop_to_process.src[1]), arg=current_uop_to_process.arg)
      print(f"  Retargeted ASSIGN for acc {uop_in_body.src[0].arg if uop_in_body.src[0].arg else id(uop_in_body.src[0])} to new target {new_target_acc.arg if new_target_acc.arg else id(new_target_acc)}")

    if current_uop_to_process == original_loop_var_uop:
        new_loop_body.append(calculated_new_idx_uop)
        # print(f"  Replaced UOp {current_uop_to_process.arg} directly with {calculated_new_idx_uop.arg}")
        continue

    # Do not substitute loopvar within the sources of a DEFINE_ACC itself.
    if current_uop_to_process.op is Ops.DEFINE_ACC:
      # If this DEFINE_ACC is an original one that should now be tile_local (i.e. it's a key in acc_target_remap),
      # it means its definition (if it was part of loop_body) is effectively replaced by the new tile_local_acc_definition
      # created in tile_loop, which is placed in tile_local_acc_inits.
      # So, we should not add the old definition from the loop_body here.
      # However, tile_local_acc_definition itself is added to tile_local_acc_inits, not directly into modified_loop_body here.
      # The ASSIGNs to it are what matter inside the loop body.
      # The original DEFINE_ACC uops from the loop_body should generally not be carried over if they were transformed.
      # This logic might need refinement based on whether DEFINE_ACC uops are ever actually *in* the loop_body list
      # or if they are always outside and only ASSIGNs are inside.
      # For now, if it's an original acc that's remapped, we assume its new definition + init is handled.
      # If it's some other DEFINE_ACC not part of the remapping, keep it.
      if current_uop_to_process not in acc_target_remap:
          new_loop_body.append(current_uop_to_process)
      else:
          print(f"  Skipping original DEFINE_ACC {current_uop_to_process.arg if current_uop_to_process.arg else id(current_uop_to_process)} from loop body as it's transformed.")
      continue

    modified_uop = current_uop_to_process.substitute(idx_substitution_map)

    # if modified_uop is not uop_in_body:
        # print(f"  UOp {uop_in_body.op} {uop_in_body.arg} was modified.")
        # if modified_uop.op is Ops.INDEX:
            # print(f"    New INDEX UOp src[1] is now: {modified_uop.src[1].op} arg={modified_uop.src[1].arg if hasattr(modified_uop.src[1], 'arg') else 'N/A'}")
            # if any(s == original_loop_var_uop for s in uop_in_body.src):
            #      print(f"    Original loop var was a direct source of the INDEX.")
            # if original_loop_var_uop in uop_in_body.src[1].toposort() if hasattr(uop_in_body.src[1], 'toposort') else False:
            #     print(f"    Original loop var was in toposort of old index src[1].")
    new_loop_body.append(modified_uop)

  # print(f"Note: _modify_loop_body_for_tiling uses UOp.substitute(). Ensure Ops.SPECIAL with ('loopvar', idx, tag) is correctly handled by it.")
  return new_loop_body, inserted_calc_uops
