import Lake
open Lake DSL

package «st_proof» where
  -- add package configuration here if needed

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib «StProof» where
  -- library configuration
