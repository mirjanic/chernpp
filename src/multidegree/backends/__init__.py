"""
Registry of multidegree algorithms.

Selecting an algorithm by name keeps the pipeline open: a new route to the
multidegree -- or a singularity family beyond Morin A_d -- is added by writing a
:class:`~multidegree.backends.base.MultidegreeBackend` and registering it, with
no change to the chamber assembly, the artifact schema, or anything in
:mod:`chernpp`.

To add one::

    from .base import Multidegree, MultidegreeBackend

    class MyBackend(MultidegreeBackend):
        name = "my-backend"
        families = ("morin-a",)
        description = "one line, shown by --list-backends"

        def compute(self, family, order, base_field):
            ...
            return Multidegree(polynomial=Q, ring=R, family=family,
                               order=order, codim=expected)

then add it to ``_BUILTIN`` below, or call :func:`register` from your own code.
Selection happens via ``--backend`` on :mod:`multidegree.build`.
"""

from typing import Dict, List

from .base import Multidegree, MultidegreeBackend

#: Import paths of the backends shipped with Chern++, as (module, class-name).
#: Imported lazily so that listing does not pay for every backend's dependencies.
#:
#: Two further routes are designed but unimplemented, each with its rationale and
#: the work required written up in its module:
#:
#:   * :mod:`multidegree.backends.bott_samelson` -- resolve the orbit closure and
#:     push forward by localisation.  Would slot in here directly, and is the most
#:     plausible route to d = 7.
#:   * :mod:`multidegree.backends.restriction` -- Rimányi's restriction equations.
#:     Produces a Thom polynomial rather than a multidegree, so it belongs in the
#:     pipeline as an independent cross-check rather than as a backend.
_BUILTIN = [("multidegree.backends.basic_equations", "BasicEquationsBackend")]

_REGISTRY: Dict[str, MultidegreeBackend] = {}

#: Built-ins that could not be imported, and why.  A backend whose dependencies
#: are absent -- SageMath, most obviously -- must not take the registry down
#: with it, so import failures are recorded and surfaced on lookup instead.
_UNAVAILABLE: Dict[str, str] = {}


def register(backend: MultidegreeBackend, replace: bool = False) -> MultidegreeBackend:
    """Make ``backend`` selectable by its :attr:`~MultidegreeBackend.name`."""
    if not backend.name:
        raise ValueError("a backend must define a non-empty name")
    if backend.name in _REGISTRY and not replace:
        raise ValueError(f"backend {backend.name!r} is already registered")
    _REGISTRY[backend.name] = backend
    return backend


def _load_builtins() -> None:
    from importlib import import_module

    for module_path, class_name in _BUILTIN:
        if module_path in _UNAVAILABLE:
            continue
        try:
            backend_class = getattr(import_module(module_path), class_name)
        except Exception as exc:  # missing SageMath, most likely
            _UNAVAILABLE[module_path] = f"{type(exc).__name__}: {exc}"
            continue
        if backend_class.name not in _REGISTRY:
            register(backend_class())


def unavailable() -> Dict[str, str]:
    """Built-in backends that failed to import, mapped to the reason."""
    _load_builtins()
    return dict(_UNAVAILABLE)


def available() -> List[MultidegreeBackend]:
    """Every registered backend, in name order."""
    _load_builtins()
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def names() -> List[str]:
    return [backend.name for backend in available()]


def get(name: str) -> MultidegreeBackend:
    """Look up a backend by name, with a helpful error if it is unknown."""
    _load_builtins()
    if name in _REGISTRY:
        return _REGISTRY[name]
    detail = f"available: {', '.join(sorted(_REGISTRY)) or 'none'}"
    if _UNAVAILABLE:
        reasons = "; ".join(f"{mod} ({why})" for mod, why in _UNAVAILABLE.items())
        detail += f". Some backends could not be imported: {reasons}"
    raise KeyError(f"unknown backend {name!r}; {detail}")


def for_family(family: str) -> List[MultidegreeBackend]:
    """Backends that declare support for ``family``."""
    return [backend for backend in available() if backend.supports(family)]


def families() -> List[str]:
    """Every singularity family some registered backend can handle."""
    return sorted({family for backend in available() for family in backend.families})


__all__ = [
    "Multidegree",
    "MultidegreeBackend",
    "available",
    "families",
    "for_family",
    "get",
    "names",
    "unavailable",
    "register",
]
