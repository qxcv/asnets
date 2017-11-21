(define (problem BLOCKS-6-2-REVERSE)
(:domain BLOCKS)
(:objects E F B D C A - block)
;; goal is to remove thing from bottom of the stack and put it on top
(:INIT (CLEAR F) (ONTABLE A) (ON F B) (ON B C) (ON C D) (ON D E) (ON E A)
 (HANDEMPTY))
(:goal (AND (ON A B) (ON B C) (ON C D) (ON D E) (ON E F)))
)
