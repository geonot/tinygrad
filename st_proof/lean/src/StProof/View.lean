import Mathlib

namespace StProof

/-- A minimal arithmetic view: a strided arithmetic progression between two bounds. -/
structure View where
  starting : Nat
  ending   : Nat
  step     : Nat
  step_pos : 0 < step
  deriving Repr

/-- Membership of an index in a view. -/
def View.contains (v : View) (e : Nat) : Prop :=
  (v.starting ≤ e ∧ e ≤ v.ending) ∧ v.step ∣ (e - v.starting)

/-- A merged view covers exactly the union of the two input views. -/
def IsCorrectlyMerged (v₁ v₂ v₃ : View) : Prop :=
  ∀ x, v₁.contains x ∨ v₂.contains x ↔ v₃.contains x

/-- Mergeability: existence of a merged view. -/
def IsMergeable (v₁ v₂ : View) : Prop := ∃ v₃, IsCorrectlyMerged v₁ v₂ v₃

@[simp]
lemma contains_subsingleton (v : View) (e : Nat) : Decidable (v.contains e) := inferInstance

/-- Symmetry of correct merge (commutativity of union). -/
lemma IsCorrectlyMerged.symm {v₁ v₂ v₃ : View}
  (h : IsCorrectlyMerged v₁ v₂ v₃) : IsCorrectlyMerged v₂ v₁ v₃ := by
  intro x; simpa [Or.comm] using (h x)

/-- The merged step divides the gcd of steps (to be proven). -/
lemma IsCorrectlyMerged.divides_step
    {v₁ v₂ v₃ : View} (h : IsCorrectlyMerged v₁ v₂ v₃) : v₃.step ∣ Nat.gcd v₁.step v₂.step := by
  -- TODO: complete proof by counting arguments over residues modulo steps
  sorry

end StProof
