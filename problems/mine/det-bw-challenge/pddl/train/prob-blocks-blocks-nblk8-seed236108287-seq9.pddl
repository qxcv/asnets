(define (problem blocks-nblk8-seed236108287-seq9)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 - block)
    (:init (handempty) (on b1 b7) (on b2 b3) (ontable b3) (ontable b4) (on b5 b8) (on b6 b2) (on b7 b5) (on b8 b4) (clear b1) (clear b6))
    (:goal (and (handempty) (ontable b1) (on b2 b1) (on b3 b4) (ontable b4) (on b5 b8) (on b6 b3) (on b7 b5) (on b8 b2) (clear b6) (clear b7))))