;; -*-lisp-*-
;; Search & Rescue domain:
;;
;;   Florent Teichteil and Patrick Fabiani, 2008

(define (domain search-and-rescue)
  
  (:requirements :typing :equality :negative-preconditions
    :disjunctive-preconditions :universal-preconditions
    :conditional-effects :mdp)
  
  (:types zone)
  
  (:constants base - zone)
  
  (:predicates
    (at ?loc - zone)
    (explored ?loc - zone)
    (landable ?loc - zone)
    (human-onboard)
    (human-alive)
    (human-rescued)
    (on-ground)
    (mission-ended)
  )
  
  (:action goto
    :parameters (?loc - zone)
    :precondition (and
      (not (on-ground))
      (not (at ?loc))
      (imply (not (= ?loc base))
        (human-alive)
      )
    )
    :effect (and
      (forall (?prev-loc - zone)
        (when (at ?prev-loc)
	  (not (at ?prev-loc))
	)
      )
      (at ?loc)
      (when (and
        (human-onboard)
	(human-alive)
        )
        (probabilistic
          0.05 (and
	    (not (human-alive))
	    (decrease (reward) 1000)
	  )
        )
      )
    )
  )
  
  (:action explore
    :parameters (?loc - zone)
    :precondition (and
      (not (= ?loc base))
      (at ?loc)
      (not (on-ground))
      (human-alive)
    )
    :effect (and
      (explored ?loc)
      (probabilistic
        0.7 (landable ?loc)
	0.3 (not (landable ?loc))
      )
    )
  )
  
  (:action land
    :parameters (?loc - zone)
    :precondition (and
      (at ?loc)
      (not (on-ground))
      (imply (not (= ?loc base))
        (and
	  (human-alive)
	  (landable ?loc)
	)
      )
    )
    :effect (and
      (on-ground)
      (when (not (= ?loc base))
        (not (landable ?loc))
      )
      (when (human-alive)
        (and
          (when (and
	      (human-onboard)
	      (= ?loc base)
	    )
	    (and
	      (increase (reward) 1000)
	      (human-rescued)
	    )
	  )
	  (when (and
	      (not (human-onboard))
	      (not (= ?loc base))
	    )
	    (probabilistic
              0.2 (and
	        (not (human-alive))
	        (decrease (reward) 1000)
	      )
	      0.8 (human-onboard)
            )
	  )
	)
      )
    )
  )
  
  (:action takeoff
    :parameters (?loc -zone)
    :precondition (and
      (on-ground)
      (at ?loc)
      (imply (= ?loc base)
        (and
	  (human-alive)
	  (not (human-rescued))
	)
      )
    )
    :effect (not (on-ground))
  )
  
  (:action end-mission
    :precondition (and
        (at base)
	(or
	  (human-rescued)
	  (not (human-alive))
	)
      )
    :effect (mission-ended)
  )
)
