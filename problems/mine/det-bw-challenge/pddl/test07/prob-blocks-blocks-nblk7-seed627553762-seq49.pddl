(define (problem blocks-nblk7-seed627553762-seq49)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b6) (on b2 b1) (on b3 b5) (on b4 b7) (ontable b5) (on b6 b3) (ontable b7) (clear b2) (clear b4))
    (:goal (and (handempty) (on b1 b6) (ontable b2) (on b3 b5) (on b4 b7) (ontable b5) (on b6 b2) (ontable b7) (clear b1) (clear b3) (clear b4))))