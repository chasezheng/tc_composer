import os

import tensor_comprehensions as tc
import torch

from tc_composer.func.activation import Activation
from tc_composer.func.function_with_params import OptionNotFound
from ..torch_test_case import TorchTestCase


class TestFuncWithParams(TorchTestCase):
    def setUp(self):
        class TestActivation(Activation):
            # Like a copy of the class
            pass

        self.activation = TestActivation('tanh')
        self.inp = torch.randn(50)

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
