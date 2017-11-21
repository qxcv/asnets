import os
# hacky, but whatever
import sys
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

import mdpsim  # noqa: #402
import pytest  # noqa: E402
import tensorflow as tf  # noqa: E402
import numpy as np  # noqa: E402

pytest.register_assert_rewrite('models')
from models import check_prob_dom_meta, get_domain_meta, \
    get_problem_meta  # noqa: E402
from tf_utils import masked_softmax  # noqa: E402


def test_prob_dom_meta():
    tt_path = os.path.join(my_path, 'triangle-tire.pddl')
    mdpsim.parse_file(tt_path)
    domain = mdpsim.get_domains()['triangle-tire']
    problem = mdpsim.get_problems()['triangle-tire-2']

    dom_meta = get_domain_meta(domain)
    prob_meta = get_problem_meta(problem)

    check_prob_dom_meta(prob_meta, dom_meta)


class TestMaskedSoftmax(tf.test.TestCase):
    def test_masked_softmax(self):
        with self.test_session():
            values = [[-1, 3.5, 2], [-1, 3.5, 2], [1, 0, 3]]
            mask = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
            result = masked_softmax(values, mask)

            def real_softmax(vec):
                exps = np.exp(vec)
                return exps / np.sum(exps, axis=-1, keepdims=True)

            # uniform because nothing's enabled
            row0 = [1 / 3.0, 1 / 3.0, 1 / 3.0]
            # not uniform
            row1 = real_softmax([-1, 3.5, 2])
            # not uniform, but first one doesn't count
            row2 = np.concatenate([[0], real_softmax([0, 3])])
            expected = [row0, row1, row2]

            self.assertAllClose(expected, result.eval())
