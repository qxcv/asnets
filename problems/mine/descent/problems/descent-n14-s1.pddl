(define (problem descent-n14-s1)
  (:domain descent)
  (:objects l00 l01 l02 l03 l04 l05 l06 l07 l08 l09 l10 l11 l12 l13 - location)
  (:init (at l00) (alive) (have-rope)
    (descent l00 l01)
    (descent l00 l02)
    (descent l00 l03)
    (descent l01 l02)
    (descent l02 l05)
    (descent l03 l04)
    (descent l03 l06)
    (descent l04 l05)
    (descent l05 l07)
    (descent l06 l08)
    (descent l07 l09)
    (descent l08 l10)
    (descent l08 l11)
    (descent l09 l10)
    (descent l09 l12)
    (descent l10 l12)
    (descent l11 l12)
    (descent l12 l13)
  )
  (:goal (and (at l13) (alive)))
)