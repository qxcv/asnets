(define (problem blocks-nblk5-seed896255376-seq7)
    (:domain blocks)
    (:objects b1 b2 b3 b4 b5 - block)
    (:init (handempty) (ontable b1) (ontable b2) (on b3 b4) (on b4 b5) (on b5 b1) (clear b2) (clear b3))
    (:goal (and (handempty) (on b1 b2) (on b2 b5) (ontable b3) (ontable b4) (ontable b5) (clear b1) (clear b3) (clear b4))))
