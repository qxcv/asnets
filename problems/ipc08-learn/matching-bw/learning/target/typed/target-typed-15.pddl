

(define (problem matching-bw-typed-n20-matching-bw-learning-target-typed-15)
(:domain matching-bw-typed)
(:requirements :typing)
(:objects h1 h2 - hand b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20  - block)
(:init
 (empty h1)
 (empty h2)
 (hand-positive h1)
 (hand-negative h2)
 (solid b1)
 (block-positive b1)
 (on b1 b10)
 (solid b2)
 (block-positive b2)
 (on b2 b5)
 (solid b3)
 (block-positive b3)
 (on b3 b16)
 (solid b4)
 (block-positive b4)
 (on b4 b1)
 (solid b5)
 (block-positive b5)
 (on b5 b4)
 (solid b6)
 (block-positive b6)
 (on b6 b3)
 (solid b7)
 (block-positive b7)
 (on b7 b15)
 (solid b8)
 (block-positive b8)
 (on b8 b2)
 (solid b9)
 (block-positive b9)
 (on b9 b8)
 (solid b10)
 (block-positive b10)
 (on b10 b7)
 (solid b11)
 (block-negative b11)
 (on b11 b18)
 (solid b12)
 (block-negative b12)
 (on-table b12)
 (solid b13)
 (block-negative b13)
 (on b13 b20)
 (solid b14)
 (block-negative b14)
 (on-table b14)
 (solid b15)
 (block-negative b15)
 (on b15 b14)
 (solid b16)
 (block-negative b16)
 (on b16 b17)
 (solid b17)
 (block-negative b17)
 (on-table b17)
 (solid b18)
 (block-negative b18)
 (on b18 b13)
 (solid b19)
 (block-negative b19)
 (on b19 b11)
 (solid b20)
 (block-negative b20)
 (on b20 b12)
 (clear b6)
 (clear b9)
 (clear b19)
)
(:goal
(and
 (on b1 b9)
 (on b2 b20)
 (on b3 b2)
 (on b4 b8)
 (on b5 b17)
 (on b6 b5)
 (on b7 b10)
 (on b9 b19)
 (on b10 b1)
 (on b11 b7)
 (on b12 b4)
 (on b13 b3)
 (on b15 b11)
 (on b17 b14)
 (on b18 b12)
 (on b19 b6))
)
)

