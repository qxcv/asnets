(define (problem ex-bw-train-n08-s00-r846148)
  (:domain exploding-blocksworld)
  (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
  (:init (emptyhand) (on b1 b7) (on b2 b3) (on b3 b8) (on-table b4) (on b5 b6) (on-table b6) (on b7 b2) (on b8 b5) (clear b1) (clear b4) (no-detonated b1) (no-destroyed b1) (no-detonated b2) (no-destroyed b2) (no-detonated b3) (no-destroyed b3) (no-detonated b4) (no-destroyed b4) (no-detonated b5) (no-destroyed b5) (no-detonated b6) (no-destroyed b6) (no-detonated b7) (no-destroyed b7) (no-detonated b8) (no-destroyed b8) (no-destroyed-table))
  (:goal (and (on-table b1) (on b2 b7) (on-table b3) (on-table b4) (on b5 b3) (on b6 b8) (on b7 b4) (on b8 b1)  )
)
)
