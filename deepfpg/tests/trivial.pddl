;; this is used as a test fixture

(define (domain trivial)
  (:requirements :typing :strips :equality :probabilistic-effects)
  (:predicates (make-me-true ?o))
  (:action make-one-true
           :parameters (?o)
           :precondition (and )
           :effect (and (make-me-true ?o)))
  ; I STRONGLY suspect my tf.gather hacking, transposing, etc. is causing the change in batch size
  (:action make-two-true
           :parameters (?o1 ?o2)
           ; for some reason the next line crashes things with "TruthyWrapper was never meant to be instantiated"
           ; :precondition (and (twopred))
           ; also including a proposition with no actions gives me "no relevant actions for proposition <name>"
           :effect (and (make-me-true ?o1)
                        (make-me-true ?o2))))

(define (problem trivial)
  (:domain trivial)
  (:objects o1)
  (:init )
  (:goal (and (make-me-true o1))))

(define (problem trivial-2pred)
  (:domain trivial)
  (:objects o1 o2)
  (:init )
  (:goal (and (make-me-true o1) (make-me-true o2))))
