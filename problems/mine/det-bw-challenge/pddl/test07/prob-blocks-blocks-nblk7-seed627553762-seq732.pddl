(define (problem blocks-nblk7-seed627553762-seq732)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b3) (ontable b2) (on b3 b4) (ontable b4) (on b5 b7) (ontable b6) (on b7 b2) (clear b1) (clear b5) (clear b6))
    (:goal (and (handempty) (ontable b1) (on b2 b6) (on b3 b2) (on b4 b5) (ontable b5) (on b6 b4) (on b7 b3) (clear b1) (clear b7))))