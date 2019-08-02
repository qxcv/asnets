;; variant of the cn problem in which all things are initially open (should make
;; it impossible to figure out the correct direction to travel in)
(define (problem cosanostra-n4)
  (:domain cosanostra)
  (:objects
    b0 b1 b2 b3 - toll-booth
    shop home - open-intersection)
  (:init (deliverator-at shop) (pizza-at shop) (tires-intact)
    (open b0) (open b1) (open b2) (open b3) ;; wooo! this is the dangerous part
    (road shop b0) (road b0 shop)
    (road home b3) (road b3 home)
    (road b0 b1) (road b1 b0) (road b1 b2) (road b2 b1) (road b2 b3) (road b3 b2))
  (:goal (and (pizza-at home) (deliverator-at shop))))
