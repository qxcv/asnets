(define (problem monster-n10)
  (:domain monster)
  (:objects
    left-1 left-2 left-3 left-4 left-5 left-6 left-7 left-8 left-9 right-1 right-2 right-3 right-4 right-5 right-6 right-7 right-8 right-9
    - location)
  (:init (robot-at start)
    (conn left-1 left-2) (conn left-2 left-3) (conn left-3 left-4) (conn left-4 left-5) (conn left-5 left-6) (conn left-6 left-7) (conn left-7 left-8) (conn left-8 left-9) (conn left-9 left-end) (conn right-1 right-2) (conn right-2 right-3) (conn right-3 right-4) (conn right-4 right-5) (conn right-5 right-6) (conn right-6 right-7) (conn right-7 right-8) (conn right-8 right-9) (conn right-9 right-end) (conn start left-1) (conn start right-1) (conn left-end finish) (conn right-end finish)
   )
  (:goal (and (robot-at finish)))
)
