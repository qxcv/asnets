;; System Administrator Problem, Variant of (Guestrin, Koller, Parr; IJCAI-2001)
;; SSP version by Scott Sanner (ssanner@gmail.com)
;; SLP version by Olivier Buffet (firstname.lastname@loria.fr)
;;
;; Note: The original SysAdmin is discounted infinite horizon, with an additive
;;       reward function, and a transition function probability that scales
;;       according to the number of connected computers that are "up".
;;       The latter two additive aspects cannot be encoded in a lifted manner 
;;       in PPDDL.
;;
;;       In the SSP version, a computer may fail if at least one of
;;       its upstream connections has failed, so it is important to
;;       reboot the computers with the highest downstream impact
;;       first.
;;
;;       This is the SLP version. The goal is to survive with at least
;;       one running server for as long as possible. You get a unit
;;       reward for each running server at each time step.

(define (domain sysadmin-slp)

 (:requirements :typing :equality :disjunctive-preconditions
                :probabilistic-effects :existential-preconditions
                :conditional-effects :negative-preconditions
                :universal-preconditions :rewards
		:equality)
 (:types comp)

 (:predicates (up ?c)
              (conn ?c ?d))

;; Note: MDPSim performs an action by going down its effect tree,
;; storing adds and dels (of terms) and updates (of values). The
;; "reboot" action has been modified to prevent the term (up ?x) from
;; being in both the adds and dels lists.
;; All tests are based on old terms and values (which is good). Terms
;; and values are updated only after the "action tree" has been
;; processed. As a result, no chain reaction can take place.
(:action reboot
  :parameters (?x - comp)
  :effect (and (forall (?d - comp)
		       (when (up ?d)
			 (increase (reward) 1)
			 )
		       )
               (probabilistic 0.9 (up ?x))
               (forall (?d - comp)
		       (and
			(probabilistic
			 0.2 (when (exists (?c - comp) (and (conn ?c ?d) (not (up ?c)) (not (= ?d ?x))))
			       (not (up ?d))
			       ))
			(probabilistic
			 0.05 (when (not (= ?d ?x))
				(not (up ?d))
			 ))
			)
		   )))
)
