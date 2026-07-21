Documentation Standards
=======================

The reference documentation is part of ngehtsim's public interface. A change
is not complete merely because its tests pass: users must be able to discover
what the changed interface accepts, produces, and deliberately does not
support.

Public APIs
-----------

Public classes, functions, methods, and configuration values must have
NumPy-style docstrings that state their parameters, return values, raised
errors, units, array shapes, and representation constraints where applicable.
For example, an uncertainty description must say whether it applies to the
complex amplitude or independently to its real and imaginary components.

The following are public for documentation purposes:

* Names presented in user tutorials, API pages, package initializers, or file
  format specifications.
* Stable native data structures and their boundary adapters.
* User-configurable simulation and weather behavior.

Private helpers do not need repetitive narrative. They do need a concise
docstring or orienting comment when their algorithm, coordinate convention, or
side effects are not evident from the code.

User Guides and CI
------------------

Docstrings are necessary but not sufficient. New data models, file formats,
and breaking changes also require a narrative page with examples and explicit
limitations. The documentation workflow builds Sphinx with warnings treated as
errors, so unresolved references and malformed directives block a pull
request. API and tutorial changes should be reviewed together with their
implementation and tests.
