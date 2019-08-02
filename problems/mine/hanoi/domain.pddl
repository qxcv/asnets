(define (domain hanoi)
  ;; The Towers of Hanoi problem (formalisation by Hector Geffner).
  (:requirements :typing :strips)
  (:types thing - object)
  (:predicates (clear ?x - thing) (on ?x ?y - thing) (smaller ?x ?y - thing))
  (:action move
    :parameters (?disc ?from ?to - thing)
    :precondition (and (smaller ?to ?disc) (on ?disc ?from)
      (clear ?disc) (clear ?to))
    :effect  (and (clear ?from) (on ?disc ?to) (not (on ?disc ?from))
      (not (clear ?to)))))
