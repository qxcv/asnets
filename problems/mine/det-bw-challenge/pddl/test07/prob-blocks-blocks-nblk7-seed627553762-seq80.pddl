(define (problem blocks-nblk7-seed627553762-seq80)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (ontable b1) (on b2 b7) (ontable b3) (on b4 b1) (on b5 b2) (on b6 b3) (on b7 b6) (clear b4) (clear b5))
    (:goal (and (handempty) (ontable b1) (ontable b2) (ontable b3) (ontable b4) (on b5 b2) (on b6 b4) (on b7 b1) (clear b3) (clear b5) (clear b6) (clear b7))))
