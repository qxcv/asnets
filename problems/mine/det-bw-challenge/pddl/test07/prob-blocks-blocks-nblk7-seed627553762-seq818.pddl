(define (problem blocks-nblk7-seed627553762-seq818)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (ontable b1) (on b2 b5) (on b3 b4) (on b4 b6) (on b5 b3) (on b6 b7) (on b7 b1) (clear b2))
    (:goal (and (handempty) (on b1 b7) (on b2 b6) (ontable b3) (on b4 b5) (ontable b5) (on b6 b1) (ontable b7) (clear b2) (clear b3) (clear b4))))