import tensor_comprehensions as tc
import torch
from torch import Tensor, optim, nn

from tc_composer.func.affine_transform import AffineTransform
from tc_tuner.async_queue import AsyncQueue
from tc_tuner.modules import Vectorizer, Decorrelation, Evaluator, Proposer
from ..torch_test_case import TorchTestCase
import tempfile

class TestVectorizer(TorchTestCase):

    def setUp(self):
        self.options = (
            tc.MappingOptions('naive'),
            #tc.MappingOptions('naive').maxSharedMemory(103),
            tc.MappingOptions('naive').mapToThreads([103]),
            tc.MappingOptions('naive').mapToThreads([103, 99]),
            tc.MappingOptions('naive').mapToBlocks([103, 99]),
            tc.MappingOptions('naive').mapToBlocks([99]),
            tc.MappingOptions('naive').tile([99]),
            tc.MappingOptions('naive').tile([99, 103]),
            tc.MappingOptions('naive').outerScheduleFusionStrategy('Max'),
            tc.MappingOptions('naive').mapToBlocks([1000, 6, 512]).mapToThreads([6, 6]).tile(
            [16384, 48, 1024, 1, 4096]).intraTileScheduleFusionStrategy('Max').outerScheduleFusionStrategy(
            'Max').unroll(32)
        )

    def test_parse_option_str(self):
        for o in self.options:
            self.assertEqual(str(o), str(Vectorizer.from_attr_to_opt(Vectorizer.parse_option_str(str(o)))))

    def test_to_ints_from_ints(self):
        inp = tuple(range(0, 10000, 11))
        self.assert_allclose(inp, Vectorizer.to_positive_ints(Vectorizer.from_positive_ints(inp)), atol=1e-15)
        with self.assertRaises(AssertionError):
            Vectorizer.from_positive_ints((-1,))

    def test_to_class_from_class(self):
        classes = tuple(range(5))
        for i in classes:
            self.assertEqual(i, Vectorizer.to_class(Vectorizer.from_class(i, classes), classes))

    def test_from_class(self):
        for attr, mytype, length in Vectorizer.CONFIG:
            if mytype is not int:
                self.assertEqual(len(Vectorizer.from_class(mytype[0], mytype)), length)

    def test_mapping_options(self):
        for opt in self.options:
            self.assertEqual(str(opt),
                             str(Vectorizer.to_mapping_options(Vectorizer.from_mapping_options(opt))))

    def test_options_class(self):
        option_attr = Vectorizer.parse_option_str(str(tc.MappingOptions('naive')))
        queue: AsyncQueue[Vectorizer.OptionAttr] = AsyncQueue()
        queue.put(option_attr)
        option_attr = queue.get()

        aff = AffineTransform(2, 2)
        aff.recompile(Tensor(1, 2), option=Vectorizer.from_attr_to_opt(option_attr))


class TestScaledAbsDet(TorchTestCase):
    def setUp(self):
        self.m = torch.randn(3, 3)
        self.m.requires_grad = True

    def test_abs(self):
        self.assertLess(0, Decorrelation.scaled_abs_det(self.m))
        self.assertLess(0, Decorrelation.scaled_abs_det(-self.m))

    def test_scaling(self):
        self.assert_allclose(
            Decorrelation.scaled_abs_det(self.m), Decorrelation.scaled_abs_det(self.m * 2)
        )

    def test_backward(self):
        m = torch.eye(3)
        m.requires_grad = True
        Decorrelation.scaled_abs_det(m).log().neg().backward()
        self.assert_allclose(m.grad, torch.zeros(3, 3))

    def test_optim(self):
        save = Decorrelation.scaled_abs_det(self.m)
        save.log().neg().backward()
        optimizer = optim.RMSprop((self.m,))
        optimizer.step()

        self.assertLess(save.item(), Decorrelation.scaled_abs_det(self.m).item())


class TestDecorrelation(TorchTestCase):
    def setUp(self):
        self.in_n = 4
        self.out_n = 3
        self.inp = torch.randn(10, self.in_n)

        self.lin = nn.Linear(self.in_n, self.out_n)
        self.decorrelation = Decorrelation(self.out_n)

    def test_out(self):
        self.assert_allclose(self.lin(self.inp), self.decorrelation(self.lin(self.inp)))

    def test_train(self):
        optimizer = optim.RMSprop(tuple(self.lin.parameters()))
        out0 = self.decorrelation(self.lin(self.inp))
        self.decorrelation.apply_grad()
        optimizer.step()

        out1 = self.decorrelation(self.lin(self.inp))

        d0 = Decorrelation.scaled_abs_det(torch.matmul(out0.t(), out0)).item()
        d1 = Decorrelation.scaled_abs_det(torch.matmul(out1.t(), out1)).item()

        self.assertLess(d0, d1)

    def test_non_train(self):
        self.decorrelation.train(False)
        self.decorrelation(self.lin(self.inp))
        self.decorrelation.apply_grad()
        for p in self.lin.parameters():
            self.assertIsNone(p.grad)


class TestProposer(TorchTestCase):
    def setUp(self):
        pass


class TestProposerEvaluator(TorchTestCase):
    def setUp(self):
        self.in_features = 4

        self.evaluator = Evaluator()
        self.proposer = Proposer(in_features=self.in_features)

    def test_run(self):
        t, b = self.proposer(torch.randn(self.in_features))

        predicted_t = self.evaluator(t)

        (predicted_t - 100).abs().sum().backward()

        self.assertEqual(predicted_t.shape[0], self.proposer._num_proposals)

        def get_grad(mt):
            if mt.grad is None:
                return 0
            return mt.grad.abs().sum().item()

        for n, p in enumerate(self.proposer.parameters()):
            try:
                self.assertGreater(get_grad(p), 0)
            except AssertionError:
                self.logger.info(f'n={n}')
        for n, p in enumerate(self.evaluator.parameters()):
            try:
                self.assertGreater(get_grad(p), 0)
            except AssertionError:
                self.logger.info(f'n={n}')

    def test_save(self):
        t, b = self.proposer(torch.randn(self.in_features))
        with tempfile.NamedTemporaryFile() as tmp:
            self.evaluator.save(tmp.name)

            e = Evaluator()
            e.load(tmp.name)

            for p0, p1 in zip(self.evaluator.parameters(), e.parameters()):
                self.assert_allclose(p0, p1)

            self.assert_allclose(
                e(t), self.evaluator(t)
            )


        with tempfile.NamedTemporaryFile() as tmp:
            self.proposer.save(tmp.name)

            p = Proposer(self.in_features)
            p.load(tmp.name)

            for p0, p1 in zip(self.proposer.parameters(), p.parameters()):
                self.assert_allclose(p0, p1)

            inp = torch.randn(self.in_features)
            self.assert_allclose(p(inp), self.proposer(inp))