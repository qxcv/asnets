;;;; -*- Mode: SCHEME -*-
;;;;
;;;; Author: Iain Little
;;;;
;;;; This is an adaption of a domain originally created for Prottle;
;;;; this is the timeless PPDDL version.
;;;;

;;; Note: The equality preconditions may be needed to be uncommented
;;; for some planners to use this domain.  This changes the problem
;;; slightly, so take care when making comparisons if it is necessary
;;; to do this.

(define (domain maze)
  (:requirements :probabilistic-effects :typing :strips)
  (:types key - locatable person - locatable
	  red-key - key green-key - key blue-key - key
	  location)
  (:constants start - location finish - location limbo - location)
  (:predicates (at ?o - locatable ?l - location)
	       (have ?p - person ?k - key)
	       (red ?from - location ?to - location)
	       (green ?from - location ?to - location)
	       (blue ?from - location ?to - location)
	       (open ?from - location ?to - location)
	       (closed ?from - location ?to - location))
  (:action move
      :parameters (?p - person ?from - location ?to - location)
      :precondition (and (at ?p ?from) (open ?from ?to)
; 			 (not (= ?from ?to))
			 )
      :effect (and (not (at ?p ?from))
		   (probabilistic 0.95 (at ?p ?to))))
  (:action open-red
      :parameters (?p - person ?k - red-key ?from - location ?to - location)
      :precondition (and (at ?p ?from) (have ?p ?k)
			 (red ?from ?to) (closed ?from ?to)
; 			 (not (= ?from ?to))
			 )
      :effect (probabilistic
	        0.4 (and (open ?from ?to)
			 (not (closed ?from ?to)))
		0.4 (and (open ?from ?to)
			 (not (closed ?from ?to))
			 (not (have ?p ?k)))
		0.1 (not (have ?p ?k))))
  (:action open-green
      :parameters (?p - person ?k - green-key ?from - location ?to - location)
      :precondition (and (at ?p ?from) (have ?p ?k)
			 (green ?from ?to) (closed ?from ?to)
; 			 (not (= ?from ?to))
			 )
      :effect (probabilistic
	        0.4 (and (open ?from ?to)
			 (not (closed ?from ?to)))
		0.4 (and (open ?from ?to)
			 (not (closed ?from ?to))
			 (not (have ?p ?k)))
		0.1 (not (have ?p ?k))))
  (:action open-blue
      :parameters (?p - person ?k - blue-key ?from - location ?to - location)
      :precondition (and (at ?p ?from) (have ?p ?k)
			 (blue ?from ?to) (closed ?from ?to)
; 			 (not (= ?from ?to))
			 )
      :effect (probabilistic
	        0.4 (and (open ?from ?to)
			 (not (closed ?from ?to)))
		0.4 (and (open ?from ?to)
			 (not (closed ?from ?to))
			 (not (have ?p ?k)))
		0.1 (not (have ?p ?k))))
  (:action get-key
      :parameters (?p - person ?k - key ?l - location)
      :precondition (and (at ?p ?l) (at ?k ?l))
      :effect (and (not (at ?k ?l))
		   (have ?p ?k)))
  ;; Note: The following action creates a clone of the person (that
  ;; will disappear if it ever meets the original).
  (:action gamble
      :parameters (?p - person ?k - key)
      :precondition (have ?p ?k)
      :effect (and (not (have ?p ?k))
		   (probabilistic
		     0.05 (at ?p finish)
		     0.45 (at ?p start)
		     0.5 (at ?p limbo)))))

(define (problem maze-paper)
  (:domain maze)
  (:objects p1 - person
	    k1 - red-key
	    k2 - green-key
	    k3 - blue-key
	    l1 l2 l3 - location)
  (:init (at p1 start) (at k1 start) (at k2 l1) (at k3 l3)
	 (closed start l1) (closed l1 finish) (closed l1 l2)
	 (closed l2 finish) (closed start l3) (closed l3 finish)
	 (open start l3) (open l1 start) (open l2 l1) (open l3 start)
	 (red start l1) (red l1 finish) (green l1 l2) (green l2 finish)
	 (green l3 finish) (blue l3 finish))
  (:goal (at p1 finish)))
