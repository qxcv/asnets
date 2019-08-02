(define (problem monster-n4)
  (:domain monster)
  (:objects
    left-1 left-2 left-3 right-1 right-2 right-3
    - location)
  (:init (robot-at start)
    (conn left-1 left-2) (conn left-2 left-3) (conn left-3 left-end) (conn right-1 right-2) (conn right-2 right-3) (conn right-3 right-end) (conn start left-1) (conn start right-1) (conn left-end finish) (conn right-end finish)
   )
  (:goal (and (robot-at finish)))
)
