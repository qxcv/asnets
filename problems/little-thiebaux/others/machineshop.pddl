;;;; -*- Mode: SCHEME -*-
;;;;
;;;; This domain was adapted by Iain Little from the version of
;;;; Machineshop used by Mausam and Weld in their paper "Solving
;;;; concurrent Markov decision processes" (2004).  This is the PPDDL
;;;; version.
;;;;

(define (domain MachineShop)
  (:requirements :probabilistic-effects :typing :negative-preconditions)
  (:types Piece Machine)
  (:predicates (shaped ?p - Piece)
	       (painted ?p - Piece)
	       (smooth ?p - Piece)
	       (polished ?p - Piece)
	       (canpolpaint ?m - Machine)
	       (canlatroll ?m - Machine)
	       (cangrind ?m - Machine)
	       (at ?p - Piece ?m - Machine)
	       (placed ?p - Piece)
	       (hasimmersion ?m - Machine)
	       (free ?m - Machine))
  (:action polish
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (canpolpaint ?m) (at ?x ?m) (placed ?x))
       :effect (probabilistic 0.9 (polished ?x)))
  (:action spraypaint
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (canpolpaint ?m) (at ?x ?m) (placed ?x))
       :effect (probabilistic 0.8 (painted ?x)))
  (:action immersionpaint
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (canpolpaint ?m) (hasimmersion ?m) 
			  (at ?x ?m) (placed ?x))
       :effect (and (probabilistic
		      0.38 (and (painted ?x) (not (hasimmersion ?m)))
		      0.57 (and (painted ?x))
		      0.02 (not (hasimmersion ?m)))))
  (:action lathe
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (canlatroll ?m) (at ?x ?m) (placed ?x))
       :effect (probabilistic
		 0.9 (and (shaped ?x)
			  (not (painted ?x))
			  (not (smooth ?x)))))
  (:action grind
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (cangrind ?m) (at ?x ?m) (placed ?x))
       :effect (probabilistic 0.9 (smooth ?x)))
  (:action buyimmersion
       :parameters (?m - Machine)
       :precondition (not (hasimmersion ?m))
       :effect (hasimmersion ?m))
  (:action place
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (at ?x ?m) (free ?m))
       :effect (and (not (free ?m))
		    (placed ?x)))
  (:action remove
       :parameters (?x - Piece ?m - Machine)
       :precondition (and (at ?x ?m) (placed ?x))
       :effect (and (not (placed ?x)) (free ?m)))
  (:action move
       :parameters (?x - Piece ?m1 - Machine ?m2 - Machine)
       :precondition (and (at ?x ?m1) (not (placed ?x))
			  (not (= ?m1 ?m2)))
       :effect (probabilistic
		 0.9 (and (at ?x ?m2) (not (at ?x ?m1))))))

(define (problem machineshop-paper)
  (:domain machineshop)
  (:objects x1 x2 - Piece
            m1 m2 - Machine)
  (:init (at x1 m2)
	 (at x2 m2)
	 (canpolpaint m1)
	 (canlatroll m2)
	 (cangrind m2)
	 (free m1)
	 (free m2))
  (:goal (and (shaped x1)
	      (painted x2)
	      (smooth x1)
	      (polished x2)
	      (free m1)
	      (free m2))))
