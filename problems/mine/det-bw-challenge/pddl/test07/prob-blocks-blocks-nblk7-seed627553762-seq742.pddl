(define (problem blocks-nblk7-seed627553762-seq742)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b4) (on b2 b1) (on b3 b5) (ontable b4) (on b5 b6) (ontable b6) (on b7 b3) (clear b2) (clear b7))
    (:goal (and (handempty) (on b1 b2) (on b2 b7) (ontable b3) (on b4 b6) (on b5 b3) (on b6 b1) (on b7 b5) (clear b4))))
