"""
Tier 6: the multidegree backend registry.

The backends themselves need SageMath, so what is exercised here is the
selection machinery: registration, lookup, family filtering, and the contract
that :class:`Multidegree` enforces on whatever a backend returns.  The registry
is deliberately importable without Sage, so that listing and error messages
still work when a backend's dependencies are absent.
"""

import unittest

from multidegree import backends
from multidegree.backends.base import Multidegree, MultidegreeBackend


class StubPolynomial:
    """Minimal stand-in for a Sage polynomial: it only needs a degree."""

    def __init__(self, degree):
        self._degree = degree

    def degree(self):
        return self._degree


class StubBackend(MultidegreeBackend):
    name = "stub"
    families = ("morin-a", "imaginary-family")
    description = "test double"

    def compute(self, family, order, base_field):
        return Multidegree(
            polynomial=StubPolynomial(order),
            ring=None,
            family=family,
            order=order,
            codim=order,
        )


class RegistryTestCase(unittest.TestCase):
    """Snapshot the registry so tests cannot leak into one another."""

    def setUp(self):
        self._saved = dict(backends._REGISTRY)

    def tearDown(self):
        backends._REGISTRY.clear()
        backends._REGISTRY.update(self._saved)


class TestRegistration(RegistryTestCase):
    def test_register_then_get(self):
        stub = backends.register(StubBackend())
        self.assertIs(backends.get("stub"), stub)
        self.assertIn("stub", backends.names())

    def test_duplicate_registration_is_refused(self):
        backends.register(StubBackend())
        with self.assertRaises(ValueError):
            backends.register(StubBackend())

    def test_duplicate_allowed_when_replacing(self):
        backends.register(StubBackend())
        replacement = backends.register(StubBackend(), replace=True)
        self.assertIs(backends.get("stub"), replacement)

    def test_nameless_backend_is_refused(self):
        class Nameless(StubBackend):
            name = ""

        with self.assertRaises(ValueError):
            backends.register(Nameless())

    def test_names_are_sorted(self):
        backends.register(StubBackend())
        self.assertEqual(backends.names(), sorted(backends.names()))


class TestLookupErrors(RegistryTestCase):
    def test_unknown_backend_lists_what_is_available(self):
        backends.register(StubBackend())
        with self.assertRaises(KeyError) as caught:
            backends.get("no-such-backend")
        self.assertIn("stub", str(caught.exception))

    def test_import_failures_are_surfaced_not_swallowed(self):
        # Without SageMath the built-in backend cannot import.  That must not
        # break the registry, and the reason must be reported rather than lost.
        for module, reason in backends.unavailable().items():
            with self.subTest(module=module):
                self.assertTrue(reason, "an unavailable backend must record why")


class TestFamilies(RegistryTestCase):
    def test_supports(self):
        stub = StubBackend()
        self.assertTrue(stub.supports("morin-a"))
        self.assertFalse(stub.supports("nonexistent"))

    def test_for_family_filters(self):
        backends.register(StubBackend())
        self.assertIn("stub", [b.name for b in backends.for_family("imaginary-family")])
        self.assertEqual(backends.for_family("nonexistent"), [])

    def test_families_are_collected_across_backends(self):
        backends.register(StubBackend())
        self.assertIn("imaginary-family", backends.families())


class TestMultidegreeContract(RegistryTestCase):
    def test_declared_codimension_must_match_the_polynomial(self):
        # A backend returning a multidegree of the wrong degree is a bug that
        # must surface at the boundary, not silently corrupt the artifact.
        with self.assertRaises(RuntimeError):
            Multidegree(
                polynomial=StubPolynomial(5),
                ring=None,
                family="morin-a",
                order=6,
                codim=7,
            )

    def test_consistent_result_is_accepted(self):
        result = StubBackend().compute("morin-a", 6, base_field=None)
        self.assertEqual(result.order, 6)
        self.assertEqual(result.codim, 6)
        self.assertEqual(result.family, "morin-a")

    def test_backend_refuses_unsupported_family(self):
        from multidegree.backends.base import MultidegreeBackend as Base

        class Picky(Base):
            name = "picky"
            families = ("morin-a",)
            description = ""

            def compute(self, family, order, base_field):
                if not self.supports(family):
                    raise ValueError(family)
                return None

        with self.assertRaises(ValueError):
            Picky().compute("nonexistent", 4, None)


if __name__ == "__main__":
    unittest.main()
