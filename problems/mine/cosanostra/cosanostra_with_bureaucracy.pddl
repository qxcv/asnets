;; rough summary: you work for CosaNostra Pizza, and have to take a pizza from a
;; shop to someone's home and back. On the way, you go through a bunch of
;; toll-booths: you can either pay the operator and part ways happily, or not pay
;; and save a bit of time. If you don't pay, they get angry, and they might slash
;; your tires when you pass through on the way back. Hence, it's a good idea to
;; pay the first time, but not the second :-)
(define (domain cosanostra)
  (:requirements :typing :strips :probabilistic-effects :conditional-effects :negative-preconditions)

  (:types location - object
          toll-booth open-intersection - location
          b-step)

  (:predicates (have-pizza) (tires-intact)
               (deliverator-at ?l - location) (pizza-at ?l - location)
               (open ?booth - toll-booth) (operator-angry ?booth - toll-booth)
               (road ?from ?to - location)
               (bureaucracy ?s - b-step)
               (b_next ?s - b-step ?t - b-step)
  )

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

  ;; (:action leave-toll-booth-angry
  ;;          :parameters (?from - toll-booth ?to - location)
  ;;          :precondition (and (deliverator-at ?from) (tires-intact)
  ;;                             (road ?from ?to) (operator-angry ?from))
  ;;          :effect (and (probabilistic
  ;;                        0.5 (and (not (tires-intact)))
  ;;                        0.5 (and (not (deliverator-at ?from)) (deliverator-at ?to)))))

  ;; (:action leave-toll-booth-sedate
  ;;          :parameters (?from - toll-booth ?to - location)
  ;;          :precondition (and (deliverator-at ?from) (tires-intact)
  ;;                             (road ?from ?to) (not (operator-angry ?from)))
  ;;          :effect (and (not (deliverator-at ?from)) (deliverator-at ?to)
  ;;                       (when (and (not (open ?from)))
  ;;                             (and (operator-angry ?from)))))

  (:action advance-bureaucracy
           :parameters (?s - b-step ?t - b-step)
           :precondition (and (b_next ?s ?t) (bureaucracy ?s))
           :effect (and (not (bureaucracy ?s)) (bureaucracy ?t))
  )

  (:action leave-toll-booth
           :parameters (?from - toll-booth ?to - location)
           :precondition (and (deliverator-at ?from) (tires-intact) (road ?from ?to)
                              (bureaucracy sG))
           :effect (and
                    ;; angry operators might slash your tires (even if you pay
                    ;; them---they're really quite spiteful)
                    (not (bureaucracy sG))
                    (bureaucracy s0)
                    (when (and (operator-angry ?from))
                          (and (probabilistic
                                ;; 50% chance they react in time to slash your tires
                                0.5 (and (not (tires-intact)))
                                ;; 50% chance you get through
                                0.5 (and (not (deliverator-at ?from)) (deliverator-at ?to)))))
                    ;; happy operators don't do that, though :-)
                    (when (and (not (operator-angry ?from)))
                          (and (not (deliverator-at ?from)) (deliverator-at ?to)))
                    ;; leaving without paying makes the operator mad at you when
                    ;; you come back :(
                    (when (and (not (open ?from)))
                          (and (operator-angry ?from)))))

  (:action leave-open-intersection
           ;; no obstacles at these intersections
           :parameters (?from - open-intersection ?to - location)
           :precondition (and (deliverator-at ?from) (tires-intact) (road ?from ?to))
           :effect (and (not (deliverator-at ?from)) (deliverator-at ?to))))
