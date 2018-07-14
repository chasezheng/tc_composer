import os

import tensor_comprehensions as tc
import torch
from torch import nn

from tc_composer.func.activation import Activation, Softmax
from tc_composer.func.affine_transform import AffineTransform
from tc_composer.func.function_with_params import Composition, Branch, OptionNotFound
from ..torch_test_case import TorchTestCase


class TestFuncWithParams(TorchTestCase):
    def setUp(self):
        class TestActivation(Activation):
            # Like a copy of the class
            pass

        in_n = 50
        self.activation = TestActivation('tanh', input_dim=2)
        self.activation2 = TestActivation('tanh', input_dim=2)
        self.inp = torch.randn(1, in_n)

        if os.path.exists(self.activation.option_file):
            os.remove(self.activation.option_file)

    def tearDown(self):
        if os.path.exists(self.activation.option_file):
            os.remove(self.activation.option_file)

    def test_recompile(self):
        with self.assertRaises(Exception):
            self.activation(self.inp)

        with self.assertLogs(self.activation.logger):
            self.activation.recompile(self.inp)

        self.assertIsNotNone(self.activation(self.inp))

    def test_recompile_with_option(self):
        with self.assertRaises(Exception):
            self.activation(self.inp)

        option = tc.MappingOptions('naive')
        with self.assertLogs(self.activation.logger):
            self.activation.recompile(self.inp, option=option)

        self.assertIsNotNone(self.activation(self.inp))

    def test_tune(self):
        self.assertTrue(not os.path.exists(self.activation.option_file))
        self.activation.tune_options(
            [self.inp],
            tuner_config=tc.TunerConfig().number_elites(1).generations(1).pop_size(2)
        )
        self.assertTrue(os.path.exists(self.activation.option_file))

    def test_tune_no_save(self):
        self.assertTrue(not os.path.exists(self.activation.option_file))
        self.activation.tune_options(
            [self.inp],
            tuner_config=tc.TunerConfig().number_elites(1).generations(1).pop_size(2),
            save_result=False
        )
        self.assertTrue(not os.path.exists(self.activation.option_file))

    def test_get_option(self):
        self.assertTrue(not os.path.exists(self.activation.option_file))
        with self.assertRaises(OptionNotFound):
            self.activation.get_options(self.inp, error_if_empty=True)
        option = self.activation.tune_options(
            [self.inp],
            tuner_config=tc.TunerConfig().number_elites(1).generations(1).pop_size(1)
        )

        # Each uses unique names
        self.assertNotEqual(self.activation.tc_def(self.inp),
                            self.activation2.tc_def(self.inp))
        self.assertEqual(self.activation.get_options(self.inp).serialize(),
                         self.activation2.get_options(self.inp).serialize())

        # Loaded option equals tuned option
        loaded = self.activation.get_options(self.inp, error_if_empty=True)
        self.assertEqual(option.serialize(), loaded.serialize())


class TestComposition(TorchTestCase):
    def setUp(self):
        batch_size = 2
        in_n = 3
        hidden = 5
        out_n = 7

        self.inp = torch.randn(batch_size, in_n)

        self.affine0 = AffineTransform(in_n=in_n, out_n=hidden)
        self.softmax = Softmax(input_dim=2)
        self.affine1 = AffineTransform(in_n=hidden, out_n=out_n)
        self.relu = Activation(input_dim=2, func='relu')

        self.comp = Composition(self.affine0, self.softmax, self.affine1, self.relu)
        self.torch = nn.Sequential(
            nn.Linear(in_features=in_n, out_features=hidden),
            nn.Softmax(dim=-1),
            nn.Linear(in_features=hidden, out_features=out_n),
            nn.ReLU()
        )
        for p, (_, t) in zip(self.torch.parameters(), self.comp.named_params):
            p.data = t.detach().view_as(p)

    def test_run(self):
        self.comp.recompile(self.inp)
        self.assert_allclose(
            actual=self.comp(self.inp),
            desired=self.torch(self.inp))

    def test_tc_def(self):
        self.assertIsNotNone(self.comp.tc_def(self.inp))

    def test_lshift(self):
        lshift = self.affine0 << self.softmax << self.affine1 << self.relu
        lshift.recompile(self.inp)
        self.assert_allclose(actual=lshift(self.inp), desired=self.torch(self.inp))

    def test_rshift(self):
        rshift = self.relu >> self.affine1 >> self.softmax >> self.affine0
        rshift.recompile(self.inp)
        self.assert_allclose(actual=rshift(self.inp), desired=self.torch(self.inp))

    def test_nested(self):
        lshift = self.affine0 << (self.softmax << self.affine1) << self.relu
        lshift.recompile(self.inp)
        self.assert_allclose(actual=lshift(self.inp), desired=self.torch(self.inp))

    def test_error(self):
        with self.assertRaises(AssertionError):
            # Should pass in unique functions
            Composition(self.affine0, self.affine0)


class TestBranch(TorchTestCase):
    def setUp(self):
        batch_size = 2
        in_n = 3
        out_n0 = 5
        out_n1 = 7

        self.inp = torch.rand(batch_size, in_n)

        self.softmax = Softmax(input_dim=2)
        self.nn_softmax = nn.Softmax(dim=-1)

        self.affine0 = AffineTransform(in_n=in_n, out_n=out_n0)
        self.affine1 = AffineTransform(in_n=in_n, out_n=out_n1)

        self.torch0 = nn.Linear(in_features=in_n, out_features=out_n0)
        self.torch1 = nn.Linear(in_features=in_n, out_features=out_n1)

        for p, t in zip(self.torch0.parameters(), self.affine0.params):
            p.data = t.detach().view_as(p)

        for p, t in zip(self.torch1.parameters(), self.affine1.params):
            p.data = t.detach().view_as(p)

    def test_run(self):
        branch = Branch(self.affine0, self.affine1)
        branch.recompile(self.inp)
        a, b = branch(self.inp)
        self.assert_allclose(actual=a, desired=self.torch0(self.inp))
        self.assert_allclose(actual=b, desired=self.torch1(self.inp))

    def test_add(self):
        added = self.affine0 + self.affine1

        added.recompile(self.inp)

        a, b = added(self.inp)
        self.assert_allclose(actual=a, desired=self.torch0(self.inp))
        self.assert_allclose(actual=b, desired=self.torch1(self.inp))

    def test_associativity(self):
        added0 = (self.softmax + self.affine0) + self.affine1
        added1 = self.softmax + (self.affine0 + self.affine1)
        added0.recompile(self.inp)
        added1.recompile(self.inp)

        for x,y in zip(added0(self.inp), added1(self.inp)):
            self.assert_allclose(actual=x, desired=y)

    def test_commutivity(self):
        added0 = self.affine0 + self.affine1
        added1 = self.affine1 + self.affine0
        added0.recompile(self.inp)
        added1.recompile(self.inp)
        a, b = added0(self.inp)
        a1, b1 = reversed(added1(self.inp))
        self.assert_allclose(actual=a1, desired=a)
        self.assert_allclose(actual=b1, desired=b)

    def test_compose_branch(self):
        func = Composition(self.softmax, Branch(self.affine0, self.affine1))
        self.logger.info(func.tc_def(self.inp))
        func.recompile(self.inp)

        a, b = func(self.inp)
        self.assert_allclose(actual=a, desired=self.torch0(self.nn_softmax(self.inp)))
        self.assert_allclose(actual=b, desired=self.torch1(self.nn_softmax(self.inp)))

    def test_error(self):
        with self.assertRaises(AssertionError):
            # Should pass in unique functions
            Branch(self.affine0, self.affine0)
