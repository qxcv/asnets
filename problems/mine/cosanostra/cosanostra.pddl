(define (domain cosanostra)
  (:requirements :typing :strips :probabilistic-effects :conditional-effects
                 :negative-preconditions)

  (:types location - object
          toll-booth open-intersection - location)

  (:predicates (have-pizza) (tires-intact)
               (deliverator-at ?l - location) (pizza-at ?l - location)
               (open ?booth - toll-booth) (operator-angry ?booth - toll-booth)
               (road ?from ?to - location))

  (:action load-pizza
           :parameters (?loc - location)
           :precondition (and (deliverator-at ?loc) (pizza-at ?loc))
           :effect (and (not (pizza-at ?loc)) (have-pizza)))

  (:action unload-pizza
           :parameters (?loc - location)
           :precondition (and (deliverator-at ?loc) (have-pizza))
           :effect (and (pizza-at ?loc) (not (have-pizza))))

  (:action pay-operator
           :parameters (?booth - toll-booth)
           :precondition (and (deliverator-at ?booth))
           :effect (and (open ?booth)))

  (:action leave-toll-booth
           :parameters (?from - toll-booth ?to - location)
           :precondition (and (deliverator-at ?from) (tires-intact)
                              (road ?from ?to))
           :effect (and
                    ;; angry operators might crush your car (even if you pay
                    ;; them---they're really quite spiteful)
                    (when (and (operator-angry ?from))
                          (and (probabilistic
                                ;; 50% chance they react in time to drop the
                                ;; boom
                                0.5 (and (not (tires-intact)))
                                ;; 50% chance you get through
                                0.5 (and (not (deliverator-at ?from))
                                         (deliverator-at ?to)))))
                    ;; happy operators don't do that, though
                    (when (and (not (operator-angry ?from)))
                          (and (not (deliverator-at ?from))
                               (deliverator-at ?to)))
                    ;; leaving without paying makes the operator mad at you when
                    ;; you come back
                    (when (and (not (open ?from)))
                          (and (operator-angry ?from)))))

  (:action leave-open-intersection
           ;; no obstacles at these intersections
           :parameters (?from - open-intersection ?to - location)
           :precondition (and (deliverator-at ?from) (tires-intact)
                              (road ?from ?to))
           :effect (and (not (deliverator-at ?from)) (deliverator-at ?to))))
