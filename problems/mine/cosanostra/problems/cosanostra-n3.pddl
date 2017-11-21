(define (problem cosanostra-n3)
  (:domain cosanostra)
  (:objects 
    b0 b1 b2 - toll-booth
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b2) (road b2 home)
    (road b0 b1) (road b1 b0) (road b1 b2) (road b2 b1))
  (:goal (and (pizza-at home) (deliverator-at shop))))
