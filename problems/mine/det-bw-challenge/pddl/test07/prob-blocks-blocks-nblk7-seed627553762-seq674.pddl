(define (problem blocks-nblk7-seed627553762-seq674)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b5) (on b2 b7) (on b3 b2) (ontable b4) (on b5 b6) (on b6 b4) (on b7 b1) (clear b3))
    (:goal (and (handempty) (on b1 b6) (ontable b2) (ontable b3) (on b4 b1) (on b5 b7) (on b6 b5) (ontable b7) (clear b2) (clear b3) (clear b4))))
