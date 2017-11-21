;; Problems for this domain have start location, a goal location, and two paths
;; of n transitions leading between them (both unidirectional). One path has a
;; monster at the end of it with a 99% chance of killing the agent, while the
;; other does not. You need to be able to see the monster---for this
;; probabilistic version of the domain, LM-cut landmarks produced from a
;; determinisation will not help, since the determinisation assumes the agent
;; can simply choose not to get killed by the monster if the agent runs into it.
(define (domain monster)
  (:requirements :typing :strips :probabilistic-effects :conditional-effects
                 :negative-preconditions)

  (:types location - object)

  (:constants start finish left-end right-end - location)

  (:predicates (robot-at ?l - location) (has-monster ?l - location)
               (conn ?from ?to - location) (initialised))

  ;; This action spawns a monster at a random location. The ASNet is able to
  ;; observe where the monster spawns (recall all states are fully observable),
  ;; but cannot always make use of that information due to its limited receptive
  ;; field.
  (:action init-monster
           :parameters ()
           :precondition (and (not (initialised)))
           :effect (and (initialised)
                        (probabilistic
                         1/2 (and (has-monster left-end))
                         1/2 (and (has-monster right-end)))))

  (:action drive
           :parameters (?from ?to - location)
           :precondition (and (conn ?from ?to) (robot-at ?from) (initialised))
           :effect (and
             (not (robot-at ?from))
             (when (and (has-monster ?from))
                   ;; only a 1% chance of getting to your destination
                   ;; otherwise, reach a dead end
                   (probabilistic 0.01 (robot-at ?to)))
             (when (and (not (has-monster ?from)))
                   ;; get there with certainty
                   (robot-at ?to)))))
