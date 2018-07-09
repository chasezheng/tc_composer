import os

import tensor_comprehensions as tc
import torch
from torch import nn
from tc_composer.func.activation import Activation, Softmax
from tc_composer.func.affine_transform import AffineTransform
from tc_composer.func.function_with_params import OptionNotFound
from ..torch_test_case import TorchTestCase
from tc_composer.func.function_with_params import Composition


class TestFuncWithParams(TorchTestCase):
    def setUp(self):
        class TestActivation(Activation):
            # Like a copy of the class
            pass

        self.activation = TestActivation('tanh')
        self.inp = torch.randn(1, 50)

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
            tuner_config=tc.TunerConfig().number_elites(1).generations(1).pop_size(10).mutation_rate(3)
        )
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
        self.softmax = Softmax()
        self.affine1 = AffineTransform(in_n=hidden, out_n=out_n)
        self.relu = Activation('relu')

        self.comp = Composition(self.affine0, self.softmax, self.affine1, self.relu)
        self.logger.debug('--- Definition of comp ---\n' + self.comp.tc_def)
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
        self.assertIsNotNone(self.comp.tc_def)
        self.assertNotEqual(
            self.comp.funcs[0].outs_to_keep, self.comp.funcs[1].in_names)

    def test_lshift(self):
        lshift = self.affine0 << self.softmax << self.affine1 << self.relu
        self.logger.debug('--- Definition of lshift ---\n' + lshift.tc_def)

        lshift.recompile(self.inp)
        self.assert_allclose(actual=lshift(self.inp),desired=self.torch(self.inp))

    def test_rshift(self):
        rshift = self.relu >> self.affine1 >> self.softmax >> self.affine0
        self.logger.debug('--- Definition of rshift ---\n' + rshift.tc_def)

        rshift.recompile(self.inp)
        self.assert_allclose(actual=rshift(self.inp), desired=self.torch(self.inp))