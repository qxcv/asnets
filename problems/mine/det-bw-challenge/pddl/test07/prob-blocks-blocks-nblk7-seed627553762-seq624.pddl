(define (problem blocks-nblk7-seed627553762-seq624)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b5) (ontable b2) (ontable b3) (on b4 b2) (on b5 b4) (on b6 b1) (on b7 b3) (clear b6) (clear b7))
    (:goal (and (handempty) (ontable b1) (on b2 b7) (on b3 b1) (on b4 b3) (on b5 b4) (on b6 b2) (on b7 b5) (clear b6))))
