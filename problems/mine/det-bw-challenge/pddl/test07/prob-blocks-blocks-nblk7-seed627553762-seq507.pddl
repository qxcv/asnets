(define (problem blocks-nblk7-seed627553762-seq507)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b5) (on b2 b7) (on b3 b6) (on b4 b3) (ontable b5) (on b6 b1) (on b7 b4) (clear b2))
    (:goal (and (handempty) (on b1 b7) (ontable b2) (on b3 b4) (on b4 b1) (on b5 b6) (on b6 b2) (ontable b7) (clear b3) (clear b5))))