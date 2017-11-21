;; -*-lisp-*-
;; Search & Rescue domain:
;;
;;   Florent Teichteil, 2008
;;
;; small modifications and problem generator:
;;   Olivier Buffet, 2008

(define (problem search-and-rescue-10)

  (:domain search-and-rescue)

  (:objects z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 - zone)

  (:init (at base) (on-ground) (human-alive))

  (:goal (and (mission-ended)))
  (:goal-reward 1000)
  (:metric maximize (reward))
)
