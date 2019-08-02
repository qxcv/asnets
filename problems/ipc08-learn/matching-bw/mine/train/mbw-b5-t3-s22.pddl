

(define (problem mbw-b5-t3-s22)
(:domain matching-bw-typed)
(:requirements :typing)
(:objects h1 h2 - hand b1 b2 b3 b4 b5  - block)
(:init
 (empty h1)
 (empty h2)
 (hand-positive h1)
 (hand-negative h2)
 (solid b1)
 (block-positive b1)
 (on b1 b3)
 (solid b2)
 (block-positive b2)
 (on-table b2)
 (solid b3)
 (block-negative b3)
 (on b3 b5)
 (solid b4)
 (block-negative b4)
 (on-table b4)
 (solid b5)
 (block-negative b5)
 (on-table b5)
 (clear b1)
 (clear b2)
 (clear b4)
)
(:goal
(and
 (on b3 b5)
 (on b5 b2))
)
)


