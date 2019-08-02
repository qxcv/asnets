(define (problem cosanostra-n13)
  (:domain cosanostra)
  (:objects 
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 - toll-booth
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b12) (road b12 home)
    (road b0 b1) (road b1 b0) (road b1 b2) (road b2 b1) (road b2 b3) (road b3 b2) (road b3 b4) (road b4 b3) (road b4 b5) (road b5 b4) (road b5 b6) (road b6 b5) (road b6 b7) (road b7 b6) (road b7 b8) (road b8 b7) (road b8 b9) (road b9 b8) (road b9 b10) (road b10 b9) (road b10 b11) (road b11 b10) (road b11 b12) (road b12 b11)
  )
  (:goal (and (pizza-at home) (deliverator-at shop))))
