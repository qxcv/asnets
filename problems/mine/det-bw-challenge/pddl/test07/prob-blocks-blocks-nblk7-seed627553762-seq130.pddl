(define (problem blocks-nblk7-seed627553762-seq130)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b3) (ontable b2) (on b3 b6) (on b4 b5) (on b5 b2) (on b6 b4) (ontable b7) (clear b1) (clear b7))
    (:goal (and (handempty) (on b1 b3) (ontable b2) (on b3 b6) (ontable b4) (ontable b5) (on b6 b7) (ontable b7) (clear b1) (clear b2) (clear b4) (clear b5))))