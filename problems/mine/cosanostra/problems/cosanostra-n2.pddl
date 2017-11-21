(define (problem cosanostra-n2)
  (:domain cosanostra)
  (:objects 
    b0 b1 - toll-booth
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b1) (road b1 home)
    (road b0 b1) (road b1 b0))
  (:goal (and (pizza-at home) (deliverator-at shop))))
