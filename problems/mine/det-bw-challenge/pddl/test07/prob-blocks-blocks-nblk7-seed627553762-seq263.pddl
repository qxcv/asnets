(define (problem blocks-nblk7-seed627553762-seq263)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b7) (on b2 b5) (ontable b3) (on b4 b1) (on b5 b4) (on b6 b3) (on b7 b6) (clear b2))
    (:goal (and (handempty) (on b1 b4) (ontable b2) (ontable b3) (ontable b4) (on b5 b2) (ontable b6) (on b7 b3) (clear b1) (clear b5) (clear b6) (clear b7))))
