(define (problem gripper-1)
(:domain gripper-typed)
(:objects
 rooma roomb - room
 left right - gripper
 ball1 - ball
)
(:init
(free left)
(free right)
(at ball1 rooma)
(at-robby rooma)
)
(:goal
(and
(at ball1 roomb)
)
)
)