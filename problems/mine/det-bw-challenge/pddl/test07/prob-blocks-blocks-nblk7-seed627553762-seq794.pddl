(define (problem blocks-nblk7-seed627553762-seq794)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b2) (on b2 b5) (on b3 b1) (ontable b4) (on b5 b4) (on b6 b7) (ontable b7) (clear b3) (clear b6))
    (:goal (and (handempty) (on b1 b4) (on b2 b1) (on b3 b6) (on b4 b5) (on b5 b7) (on b6 b2) (ontable b7) (clear b3))))