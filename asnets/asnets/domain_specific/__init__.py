def get_domain_specific_planner(domain_name):
    """Get a domain-specific planner for the given domain. Planners returned by
    this function are just functions that take a cstate as input & returns one
    of three things:

    (1) If state is a goal state: []
    (2) If goal cannot be reached under this generalised policy: None
    (3) Otherwise: a list of one or more action idents forming a plan prefix.

    Importantly, planners are designed such that repeatedly calling this
    planner & executing the returned plan fragment should make you hit a goal
    eventually! (except when you start in a dead end, or when your problem has
    unavoidable dead ends)"""
    if domain_name == 'gold-miner-typed':
        # lazy import b/c gold_miner depends on some other things that can
        # cause cycles
        from asnets.domain_specific import gold_miner
        return gold_miner.gold_miner_planner

    raise ValueError("No planner for '%s'" % (domain_name, ))
