(define (problem monster-n3)
  (:domain monster)
  (:objects
    left-1 left-2 right-1 right-2
    - location)
  (:init (robot-at start)
    (conn left-1 left-2) (conn left-2 left-end) (conn right-1 right-2) (conn right-2 right-end) (conn start left-1) (conn start right-1) (conn left-end finish) (conn right-end finish)
   )
  (:goal (and (robot-at finish)))
)
