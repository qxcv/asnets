

(define (problem matching-bw-untyped-n12)
(:domain matching-bw-untyped)
(:objects h1 h2 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 )
(:init
 (hand h1)
 (hand h2)
 (empty h1)
 (empty h2)
 (hand-positive h1)
 (hand-negative h2)
 (solid b1)
 (block b1)
 (block-positive b1)
 (on b1 b4)
 (solid b2)
 (block b2)
 (block-positive b2)
 (on b2 b9)
 (solid b3)
 (block b3)
 (block-positive b3)
 (on-table b3)
 (solid b4)
 (block b4)
 (block-positive b4)
 (on b4 b7)
 (solid b5)
 (block b5)
 (block-positive b5)
 (on b5 b3)
 (solid b6)
 (block b6)
 (block-positive b6)
 (on b6 b11)
 (solid b7)
 (block b7)
 (block-negative b7)
 (on b7 b2)
 (solid b8)
 (block b8)
 (block-negative b8)
 (on b8 b6)
 (solid b9)
 (block b9)
 (block-negative b9)
 (on b9 b12)
 (solid b10)
 (block b10)
 (block-negative b10)
 (on b10 b5)
 (solid b11)
 (block b11)
 (block-negative b11)
 (on b11 b1)
 (solid b12)
 (block b12)
 (block-negative b12)
 (on-table b12)
 (clear b8)
 (clear b10)
)
(:goal
(and
 (on b1 b7)
 (on b3 b8)
 (on b4 b11)
 (on b6 b1)
 (on b7 b9)
 (on b8 b5)
 (on b10 b4)
 (on b12 b6))
)
)

