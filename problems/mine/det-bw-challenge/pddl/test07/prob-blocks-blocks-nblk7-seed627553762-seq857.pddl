(define (problem blocks-nblk7-seed627553762-seq857)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b6) (ontable b2) (on b3 b7) (on b4 b5) (ontable b5) (on b6 b2) (on b7 b4) (clear b1) (clear b3))
    (:goal (and (handempty) (ontable b1) (on b2 b7) (on b3 b1) (ontable b4) (on b5 b3) (on b6 b4) (on b7 b6) (clear b2) (clear b5))))