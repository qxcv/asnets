(define (domain gripper-typed)
 (:types room gripper ball)

 (:predicates
  (at-robby ?r)
  (at ?b ?r)
  (free ?g)
  (carry ?o ?g))

 (:action move
  :parameters  (?from ?to - room)
  :precondition (and  (at-robby ?from))
  :effect (and  (at-robby ?to)
      (not (at-robby ?from))))

 (:action pick
  :parameters (?obj - ball ?room  - room ?gripper - gripper)
  :precondition  (and (at ?obj ?room) (at-robby ?room) (free ?gripper))
  :effect (and (carry ?obj ?gripper)
      (not (at ?obj ?room))
      (not (free ?gripper))))

 (:action drop
  :parameters  (?obj - ball ?room - room ?gripper - gripper)
  :precondition  (and (carry ?obj ?gripper) (at-robby ?room))
  :effect (and (at ?obj ?room)
      (free ?gripper)
      (not (carry ?obj ?gripper)))))

