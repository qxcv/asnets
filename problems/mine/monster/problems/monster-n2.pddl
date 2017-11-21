(define (problem monster-n2)
  (:domain monster)
  (:objects
    left-1 right-1
    - location)
  (:init (robot-at start)
    (conn left-1 left-end) (conn right-1 right-end) (conn start left-1) (conn start right-1) (conn left-end finish) (conn right-end finish)
   )
  (:goal (and (robot-at finish)))
)
