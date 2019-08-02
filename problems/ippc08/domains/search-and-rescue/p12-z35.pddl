;; -*-lisp-*-
;; Search & Rescue domain:
;;
;;   Florent Teichteil, 2008
;;
;; small modifications and problem generator:
;;   Olivier Buffet, 2008

(define (problem search-and-rescue-35)

  (:domain search-and-rescue)

  (:objects z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15 z16 z17 z18 z19 z20 z21 z22 z23 z24 z25 z26 z27 z28 z29 z30 z31 z32 z33 z34 z35 - zone)

  (:init (at base) (on-ground) (human-alive))

  (:goal (and (mission-ended)))
  (:goal-reward 1000)
  (:metric maximize (reward))
)
