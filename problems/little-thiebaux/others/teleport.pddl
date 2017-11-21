;;;; -*- Mode: SCHEME -*-
;;;;
;;;; Author: Iain Little
;;;;
;;;; This is an adaption of a domain originally created for Prottle;
;;;; this is the timeless PPDDL version.
;;;;

;;; Note: The equality preconditions may be needed to be uncommented
;;; for some planners to use this domain.  This changes the problem,
;;; so take care when making comparisons if it is necessary to do
;;; this.

(define (domain teleport)
  (:requirements :probabilistic-effects :typing :strips)
  (:types person location)
  (:predicates (at ?p - person ?l - location)
	       (linked ?l1 ?l2 - location) (movable ?l1 ?l2 - location))
  (:action fast-teleport
      :parameters (?p - person ?from ?to - location)
      :precondition (and (at ?p ?from) (linked ?from ?to)
; 			 (not (= ?from ?to))
			 )
      :effect (and (probabilistic
		     0.5 (and (at ?p ?to) (not (at ?p ?from)))
		     0.3 (at ?p ?from)
		     0.2 (not (at ?p ?from)))))
  (:action slow-teleport
      :parameters (?p - person ?from ?to - location)
      :precondition (and (at ?p ?from) (linked ?from ?to)
; 			 (not (= ?from ?to))
			 )
      :effect (and (not (at ?p ?from))
		   (probabilistic 0.9 (at ?p ?to))))
  (:action link
      :parameters (?source ?old-to ?new-to - location)
      :precondition (and (linked ?source ?old-to)
			 (movable ?source ?old-to)
; 			 (not (= ?old-to ?new-to))
			 )
      :effect (and (not (linked ?source ?old-to))
		   (probabilistic 0.9 (linked ?source ?new-to)))))

(define (problem teleport-paper)
  (:domain teleport)
  (:objects p1 p2 - person
	    l1 l2 l3 - location)
  (:init (linked l1 l2) (linked l3 l2) (linked l2 l1)
	 (movable l1 l2) (movable l3 l2)
	 (at p1 l1) (at p2 l3))
  (:goal (and (at p1 l3) (at p2 l1))))
