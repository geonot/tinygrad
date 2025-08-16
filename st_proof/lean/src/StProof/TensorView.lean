import Mathlib

namespace StProof

/-- Shapes are lists of positive natural numbers. -/
abbrev Shape := List Nat

/-- Contiguous row-major strides for a shape s.
    Example: [a,b,c] ↦ [b*c, c, 1]. -/
def contiguousStrides : Shape → List Nat
  | []      => []
  | [n]     => [1]
  | n :: s  => by
    let σ := contiguousStrides s
    -- product of tail shape
    let tailProd := s.foldr (· * ·) 1
    exact tailProd :: σ

/-- Dot product on lists (truncates to min length). -/
def dot : List Nat → List Nat → Nat
  | [], _ => 0
  | _, [] => 0
  | a::as, b::bs => a*b + dot as bs

/-- Unravel a linear index k to coordinates for shape s (row-major).
    Each coordinate i is (k / stride[i]) % s[i]. -/
def unravel (s : Shape) (k : Nat) : List Nat :=
  let σ := contiguousStrides s
  let coords := (List.zipWith (fun (st n : Nat) => (k / st) % n) σ s)
  coords

/-- A tensor view: a shape with strides of equal length. -/
structure TView where
  shape  : Shape
  stride : List Nat
  hlen   : shape.length = stride.length
  deriving Repr

/-- The index function of a tensor view maps coordinates to a linear offset. -/
def TView.indexFn (v : TView) (idx : List Nat) : Nat :=
  dot v.stride idx

/-- The standard contiguous view for a shape. -/
def contiguous (s : Shape) : TView :=
  { shape := s, stride := contiguousStrides s, hlen := by
      -- contiguousStrides has same length as shape
      induction s with
      | nil => simp [contiguousStrides]
      | cons n tl ih =>
        cases tl with
        | nil => simp [contiguousStrides]
        | cons n2 tl2 =>
          simp [contiguousStrides, ih] }

/-- Compose two views: v after v' by unravelling through v.shape. -/
def compose (v v' : TView) (idx' : List Nat) : Nat :=
  let k := v'.indexFn idx'
  v.indexFn (unravel v.shape k)

/-- Mergability: there exists v" such that composition equals its index function. -/
def Mergeable (v v' : TView) : Prop := ∃ v" : TView, ∀ idx', compose v v' idx' = v".indexFn idx'

/-- One-hot vector of given length with 1 at position j, 0 elsewhere. -/
def oneHot (len j : Nat) : List Nat :=
  (List.range len).map (fun i => if i = j then 1 else 0)

/-- Theoretical recovery of merge stride via one-hot probing (statement stub). -/
lemma merge_stride_recovery (v v' : TView)
  (h : Mergeable v v') : True := by
  -- TODO: formalize extraction of merged stride as composition applied to one-hots
  trivial

end StProof
