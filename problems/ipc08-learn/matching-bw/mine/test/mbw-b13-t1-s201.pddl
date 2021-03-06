

(define (problem mbw-b13-t1-s201)
(:domain matching-bw-typed)
(:requirements :typing)
(:objects h1 h2 - hand b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13  - block)
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
 (on b2 b11)
 (solid b3)
 (block-positive b3)
 (on b3 b6)
 (solid b4)
 (block-positive b4)
 (on b4 b13)
 (solid b5)
 (block-positive b5)
 (on-table b5)
 (solid b6)
 (block-positive b6)
 (on b6 b7)
 (solid b7)
 (block-negative b7)
 (on b7 b8)
 (solid b8)
 (block-negative b8)
 (on b8 b9)
 (solid b9)
 (block-negative b9)
 (on b9 b1)
 (solid b10)
 (block-negative b10)
 (on b10 b12)
 (solid b11)
 (block-negative b11)
 (on b11 b5)
 (solid b12)
 (block-negative b12)
 (on b12 b4)
 (solid b13)
 (block-negative b13)
 (on b13 b2)
 (clear b3)
)
(:goal
(and
 (on b1 b8)
 (on b2 b13)
 (on b3 b2)
 (on b4 b12)
 (on b5 b10)
 (on b6 b7)
 (on b8 b5)
 (on b9 b4)
 (on b10 b9)
 (on b11 b3)
 (on b12 b11)
 (on b13 b6))
)
)


