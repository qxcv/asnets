;;;  Author: Olivier Buffet ;;;

;;; This domain is inspired from triangle-tireworld, itself inspired
;;; by tireworld.

;;; A major difference is that there is no more spare tire to pick-up.
;;; The goal is to cross a rectangle world (usually from the
;;; bottom-left to the upper-right corner), with 8 possible moves:
;;; - 4 normal moves: U, D, R, L (uncertain if you are not on a "safe"
;;; row or column),
;;; - 4 diagonal moves: UR, UL, DR, DL (fast but dangerous moves).
;;; Any move is deadly if you leave an "unsafe" location.
;;;
;;; As in (triangle-)tireworld, deterministic planners are expected to
;;; look for short, but dangerous paths. An advantage over
;;; (triangle-)tireworld is that less variables are required to encode
;;; large problems.

(define (domain rectangle-world)
  (:requirements :typing :strips :negative-preconditions :conditional-effects :probabilistic-effects :rewards)
  (:types int)
  (:predicates (xpos ?x - int) (ypos ?y - int) (next ?i ?j - int)
	       (safeX ?x - int) (safeY ?y - int) (unsafe ?x ?y - int)
	       (dead))


  ;;; The 4 "normal" moves.
  (:action move-U
    :parameters (?x ?y ?y2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?y ?y2))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   (dead)
		   )
		 (probabilistic .8 (and (not (ypos ?y)) (ypos ?y2))
				.2 (when (safeX ?x)
				     (and (not (ypos ?y)) (ypos ?y2))
				     )
				)
		 )
    )

  (:action move-D
    :parameters (?x ?y ?y2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?y2 ?y))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   (dead)
		   )
		 (probabilistic .8 (and (not (ypos ?y)) (ypos ?y2))
				.2 (when (safeX ?x)
				     (and (not (ypos ?y)) (ypos ?y2))
				     )
				)
		 )
    )

  (:action move-R
    :parameters (?x ?y ?x2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?x ?x2))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   dead
		   )
		 (probabilistic .8 (and (not (xpos ?x)) (xpos ?x2))
				.2 (when (safeY ?y)
				     (and (not (xpos ?x)) (xpos ?x2))
				     )
				)
		 )
    )

  (:action move-L
    :parameters (?X ?y ?x2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?x2 ?x))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   dead
		   )
		 (probabilistic .8 (and (not (xpos ?x)) (xpos ?x2))
				.2 (when (safeY ?y)
				     (and (not (xpos ?x)) (xpos ?x2))
				     )
				)
		 )
    )


  ;;; The 4 diagonal moves.
  (:action move-UR
    :parameters (?x ?y ?x2 ?y2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?x ?x2) (next ?y ?y2))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   dead
		   )
		 (probabilistic .8 (and (not (xpos ?x)) (not (ypos ?y))
					(xpos ?x2) (ypos ?y2)
					)
				.2 (dead)
				)
		 )
  )

  (:action move-UL
    :parameters (?x ?y ?x2 ?y2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?x ?x2) (next ?y2 ?y))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   dead
		   )
		 (probabilistic .8 (and (not (xpos ?x)) (not (ypos ?y))
					(xpos ?x2) (ypos ?y2)
					)
				.2 (dead)
				)
		 )
  )

  (:action move-DR
    :parameters (?x ?y ?x2 ?y2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?x2 ?x) (next ?y ?y2))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   dead
		   )
		 (probabilistic .8 (and (not (xpos ?x)) (not (ypos ?y))
					(xpos ?x2) (ypos ?y2)
					)
				.2 (dead)
				)
		 )
  )

  (:action move-DL
    :parameters (?x ?y ?x2 ?y2 - int)
    :precondition (and (not (dead)) (xpos ?x) (ypos ?y) (next ?x2 ?x) (next ?y2 ?y))
    :effect (and (decrease (reward) 10)
		 (when (unsafe ?x ?y)
		   dead
		   )
		 (probabilistic .8 (and (not (xpos ?x)) (not (ypos ?y))
					(xpos ?x2) (ypos ?y2)
					)
				.2 (dead)
				)
		 )
  )

  ;;; When you're dead, you can just wander randomly. This action
  ;;; should help various planners create a bigger state space than
  ;;; necessary.
  (:action ghostTeleport
   :parameters (?X ?y ?x2 ?y2 - int)
   :precondition (and (dead) (xpos ?x) (ypos ?y))
   :effect (and
	    (decrease (reward) 1)
	    (not (xpos ?x))
	    (not (ypos ?y))
	    (xpos ?x2)
	    (ypos ?y2)
	    )
   )

)
