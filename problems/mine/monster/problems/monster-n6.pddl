(define (problem monster-n6)
  (:domain monster)
  (:objects
    left-1 left-2 left-3 left-4 left-5 right-1 right-2 right-3 right-4 right-5
    - location)
  (:init (robot-at start)
    (conn left-1 left-2) (conn left-2 left-3) (conn left-3 left-4) (conn left-4 left-5) (conn left-5 left-end) (conn right-1 right-2) (conn right-2 right-3) (conn right-3 right-4) (conn right-4 right-5) (conn right-5 right-end) (conn start left-1) (conn start right-1) (conn left-end finish) (conn right-end finish)
   )
  (:goal (and (robot-at finish)))
)
