(define (problem blocks-nblk7-seed627553762-seq103)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b4) (ontable b2) (ontable b3) (on b4 b5) (ontable b5) (on b6 b1) (on b7 b6) (clear b2) (clear b3) (clear b7))
    (:goal (and (handempty) (on b1 b6) (on b2 b5) (ontable b3) (on b4 b7) (ontable b5) (on b6 b3) (on b7 b2) (clear b1) (clear b4))))