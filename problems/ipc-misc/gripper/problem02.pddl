(define (problem gripper-2)
(:domain gripper-typed)
(:objects
 rooma roomb - room
 left right - gripper
 ball1 ball2 - ball
)
(:init
(free left)
(free right)
(at ball1 rooma)
(at ball2 rooma)
(at-robby rooma)
)
(:goal
(and
(at ball1 roomb)
(at ball2 roomb)
)
)
)