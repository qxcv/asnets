(define (problem blocks-nblk7-seed627553762-seq581)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (ontable b1) (on b2 b5) (on b3 b2) (on b4 b7) (on b5 b6) (on b6 b4) (ontable b7) (clear b1) (clear b3))
    (:goal (and (handempty) (ontable b1) (ontable b2) (on b3 b2) (on b4 b1) (on b5 b4) (on b6 b5) (ontable b7) (clear b3) (clear b6) (clear b7))))
