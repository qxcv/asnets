import re
import weakref


__all__ = [
    'HList', 'parse_sexprs', 'hlist_to_sexprs'
]


def _ppddl_tokenize(ppddl_txt):
    """Break PPDDL into tokens (brackets, non-bracket chunks)"""
    # strip comments
    lines = ppddl_txt.splitlines()
    mod_lines = []
    for line in lines:
        try:
            semi_idx = line.index(';')
        except ValueError:
            pass
        else:
            line = line[:semi_idx]
        mod_lines.append(line)
    ppddl_txt = '\n'.join(mod_lines)

    # convert to lower case
    ppddl_txt = ppddl_txt.lower()

    matches = re.findall(r'\(|\)|[^\s\(\)]+', ppddl_txt)

    return matches


def _hlist_to_tokens(hlist):
    """Convert a HList back into tokens (either single open/close parens or
    non-paren chunks)"""
    tokens = ['(']
    for item in hlist:
        if isinstance(item, HList):
            tokens.extend(_hlist_to_tokens(item))
        else:
            assert isinstance(item, str), "Can't handle item '%r'" % (item, )
            tokens.append(item)
    tokens.append(')')
    return tokens


class HList(list):
    """Class for hierarchical list. Helpful because you can get at parent from
    current node (or see that current node is root if no parent)."""

    def __init__(self, parent):
        super()
        self.is_root = parent is None
        self._parent_ref = weakref.ref(parent) if not self.is_root else None

    @property
    def parent(self):
        if self.is_root:
            return None
        return self._parent_ref()


def parse_sexprs(ppddl_txt):
    """Hacky parse of sexprs from ppddl."""
    tokens = _ppddl_tokenize(ppddl_txt)
    parse_root = parse_ptr = HList(None)
    # we parse begin -> end
    # just reverse so that pop() is efficient
    tokens_reverse = tokens[::-1]
    while tokens_reverse:
        token = tokens_reverse.pop()
        if token == '(':
            # push
            new_ptr = HList(parse_ptr)
            parse_ptr.append(new_ptr)
            parse_ptr = new_ptr
        elif token == ')':
            # pop
            parse_ptr = parse_ptr.parent
        else:
            # add
            parse_ptr.append(token)
    return parse_root


def hlist_to_sexprs(hlist):
    """Convert a HList back to (some semblance of) PDDL."""
    assert isinstance(hlist, HList), \
        "are you sure you want to pass in type %s?" % (type(hlist),)
    tok_stream = _hlist_to_tokens(hlist)

    out_parts = []
    # was the last token an open paren?
    last_open = True
    for token in tok_stream:
        is_open = token == '('
        is_close = token == ')'
        is_paren = is_open or is_close
        if (not is_paren and not last_open) or (is_open and not last_open):
            # we insert space between token seqs of the form [<non-paren>,
            # <non-paren>] and token seqs of the form [")", "("]
            out_parts.append(' ')
        out_parts.append(token)
        # for next iter
        last_open = is_open

    return ''.join(out_parts)


def extract_all_domains_problems(pddl_files):
    # make an index of domains & problems by parsing each file in turn
    domains = {}
    problems = {}
    for pddl_file in pddl_files:
        with open(pddl_file, 'r') as fp:
            pddl_txt = fp.read()
        # Each parsed file is list of domains/problems. Domain has the form:
        #
        #  ["define", ["domain", <dname>], …]
        #
        # Problem has the form:
        #
        #  ["define", ["problem", <pname>], …, [":domain", <dname>], …]
        sexprs = parse_sexprs(pddl_txt)
        for declr in sexprs:
            assert len(declr) >= 2 and declr[0] == "define", \
                "expected (define …), got AST %s" % (declr, )
            declr_type, declr_name = declr[1]
            if declr_type == "problem":
                problems[declr_name] = declr
            elif declr_type == "domain":
                domains[declr_name] = declr
            else:
                raise ValueError("Unknown type '%s'" % (declr_type,))
    return domains, problems


def extract_domain_problem(pddl_files, problem_name=None):
    """Extract HLists representing PDDL for domain & problem from a collection
    of PDDL files & a problem name."""
    domains, problems = extract_all_domains_problems(pddl_files)

    # retrieve hlist for problem & figure out corresponding domain
    if problem_name is None:
        problem_names = list(problems.keys())
        if len(problem_names) != 1:
            raise ValueError(
                "Was not given a problem name, and the given PDDL files "
                "contain either 0 or > 1 names (full list: %s)" %
                (problem_names,))
        problem_name, = problem_names
    problem_hlist = problems[problem_name]
    for subpart in problem_hlist:
        if len(subpart) == 2 and subpart[0] == ':domain':
            domain_name = subpart[1]
            break
    else:
        raise ValueError("Could not find domain for '%s'" % (problem_name, ))
    domain_hlist = domains[domain_name]

    return domain_hlist, domain_name, problem_hlist, problem_name


def extract_domain_name(pddl_path):
    """Extract a domain name from a single PDDL domain file."""
    assert isinstance(pddl_path, str), \
        "this only takes a single (string) filename"
    domains, _ = extract_all_domains_problems([pddl_path])
    assert len(domains) == 1, \
        "PDDL file at '%s' contains %d domains (not 1); they are %s" \
        % (pddl_path, len(domains), sorted(domains))
    domain_name, = domains.keys()
    return domain_name


def replace_init_state(problem_hlist, new_init_atoms):
    """Create modified hlist for problem that has old init atoms replaced with
    new set of init atoms."""
    # check format for new atoms
    assert isinstance(new_init_atoms, (tuple, list))
    for atom in new_init_atoms:
        # make sure atoms have the right format (they should all be paren-free,
        # which is the same format used when interfacing with SSiPP or MDPSim)
        assert '(' not in atom and ')' not in atom, \
            "expecting atom format with no parens, but got '%s'" % (atom, )

    # build new problem hlist
    new_hlist = HList(parent=None)
    replaced_init = False
    for subsec in problem_hlist:
        if len(subsec) >= 1 and subsec[0] == ':init':
            init_hlist = HList(parent=new_hlist)
            init_hlist.append(":init")
            init_hlist.extend('(%s)' % atom for atom in new_init_atoms)
            new_hlist.append(init_hlist)
            replaced_init = True
        else:
            new_hlist.append(subsec)

    assert replaced_init, \
        "Could not find :init in hlist '%r'" % (problem_hlist, )

    return new_hlist
