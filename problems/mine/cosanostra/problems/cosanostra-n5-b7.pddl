(define (problem cosanostra-n5-b7)
  (:domain cosanostra)
  (:objects 
    b0 b1 b2 b3 b4 - toll-booth
    s0 s1 s2 s3 s4 s5 s6 sG - b-step
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b4) (road b4 home)
    (road b0 b1) (road b1 b0) (road b1 b2) (road b2 b1) (road b2 b3) (road b3 b2) (road b3 b4) (road b4 b3)
    (b_next s6 sG)
    (b_next s0 s1) (b_next s1 s2) (b_next s2 s3) (b_next s3 s4) (b_next s4 s5) (b_next s5 s6)
    (bureaucracy s0))
  (:goal (and (pizza-at home) (deliverator-at shop))))
