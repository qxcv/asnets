"""Convenience functions for dealing with MDPSim"""


class PDDLLoadError(Exception):
    """PDDL parse exception"""


def parse_problem_args(mdpsim_module, pddls, problem_name=None):
    for pddl_path in pddls:
        success = mdpsim_module.parse_file(pddl_path)
        if not success:
            raise PDDLLoadError('Could not parse %s' % pddl_path)

    problems = mdpsim_module.get_problems()
    if problem_name is None:
        if len(problems) == 0:
            raise PDDLLoadError('Did not load any problems (?!), aborting')
        sorted_keys = sorted(problems.keys())
        problem = problems[sorted_keys[0]]
    else:
        try:
            problem = problems[problem_name]
        except KeyError:
            raise PDDLLoadError(
                'Could not find problem %s. Available problems: %s' % (
                    problem_name, ', '.join(problems.keys())))
    return problem
