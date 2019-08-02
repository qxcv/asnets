(define (problem cosanostra-n1)
  (:domain cosanostra)
  (:objects 
    b0 - toll-booth
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (road shop b0) (road b0 shop)
    (road home b0) (road b0 home)
    )
  (:goal (and (pizza-at home) (deliverator-at shop))))
