(define (problem cosanostra-n5)
  (:domain cosanostra)
  (:objects 
    b0 b1 b2 b3 b4 - toll-booth
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b4) (road b4 home)
    (road b0 b1) (road b1 b0) (road b1 b2) (road b2 b1) (road b2 b3) (road b3 b2) (road b3 b4) (road b4 b3))
  (:goal (and (pizza-at home) (deliverator-at shop))))
