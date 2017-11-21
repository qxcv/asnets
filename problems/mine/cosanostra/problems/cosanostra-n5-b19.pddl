(define (problem cosanostra-n5-b19)
  (:domain cosanostra)
  (:objects 
    b0 b1 b2 b3 b4 - toll-booth
    s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 sG - b-step
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b4) (road b4 home)
    (road b0 b1) (road b1 b0) (road b1 b2) (road b2 b1) (road b2 b3) (road b3 b2) (road b3 b4) (road b4 b3)
    (b_next s18 sG)
    (b_next s0 s1) (b_next s1 s2) (b_next s2 s3) (b_next s3 s4) (b_next s4 s5) (b_next s5 s6) (b_next s6 s7) (b_next s7 s8) (b_next s8 s9) (b_next s9 s10) (b_next s10 s11) (b_next s11 s12) (b_next s12 s13) (b_next s13 s14) (b_next s14 s15) (b_next s15 s16) (b_next s16 s17) (b_next s17 s18)
    (bureaucracy s0))
  (:goal (and (pizza-at home) (deliverator-at shop))))
