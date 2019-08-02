;; (define (problem prob_bw_4_n4_es1_r401)
;;   (:domain prob_bw)
;;   (:objects b1 b2 b3 b4 - block)
;;   (:init (emptyhand) (on b1 b3) (on-table b2) (on-table b3) (on-table b4) (clear b1) (clear b2) (clear b4))
;;   (:goal (and (emptyhand) (on b1 b4) (on-table b2) (on-table b3) (on b4 b3) (clear b1) (clear b2)))
;; )

(define (problem prob_bw_demo3)
  ;; simple reverse stack problem in prob BW
  (:domain prob_bw)
  (:objects b1 b2 b3 - block)
  (:init (emptyhand) (on-table b3) (on b2 b3) (on b1 b2) (clear b1))
  (:goal (and (emptyhand) (clear b3) (on-table b1) (on b2 b1) (on b3 b2)))
)
