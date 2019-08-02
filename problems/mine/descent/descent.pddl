(define (domain descent)
  (:requirements :typing :strips :probabilistic-effects)

  (:types location)

  (:predicates (have-rope) (alive) (at ?l - location)
               ;; indicates that we can set up a rope from ?top to ?bottom and
               ;; climb down (or just climb down!)
               (descent ?top ?bottom - location)
               ;; uh, will there be quadratically many of these? I hope not!
               ;; might have to create a "descent" object to fix this (later)
               (rope-between ?top ?bottom - location))

  (:action place-rope
           :parameters (?top ?bottom - location)
           :precondition (and (alive) (have-rope) (descent ?top ?bottom))
           :effect (and (rope-between ?top ?bottom) (not (have-rope))))

  (:action retrieve-rope
           :parameters (?top ?bottom - location)
           :precondition (and (alive) (at ?bottom) (rope-between ?top ?bottom) (descent ?top ?bottom))
           :effect (and (not (rope-between ?top ?bottom)) (have-rope)))

  (:action descend-without-rope
           :parameters (?top ?bottom - location)
           :precondition (and (alive) (at ?top) (descent ?top ?bottom))
           :effect (and (at ?bottom) (not (at ?top))
                        ;; oops
                        (probabilistic 0.8 (not (alive)))))

  (:action descend-with-rope
           :parameters (?top ?bottom - location)
           :precondition (and (alive) (at ?top) (descent ?top ?bottom) (rope-between ?top ?bottom))
           :effect (and (at ?bottom) (not (at ?top)))))
