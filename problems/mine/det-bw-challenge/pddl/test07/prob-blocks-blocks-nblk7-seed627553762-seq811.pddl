(define (problem blocks-nblk7-seed627553762-seq811)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 b6 b7 - block)
    (:init (handempty) (on b1 b4) (on b2 b5) (ontable b3) (ontable b4) (on b5 b1) (on b6 b3) (ontable b7) (clear b2) (clear b6) (clear b7))
    (:goal (and (handempty) (on b1 b2) (on b2 b4) (ontable b3) (ontable b4) (ontable b5) (ontable b6) (on b7 b1) (clear b3) (clear b5) (clear b6) (clear b7))))