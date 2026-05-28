"""Тесты для новых сущностей в `core.models`:

- :class:`core.models.CanonicalVar` — метаданные канонического столбца;
- новые опциональные поля в :class:`core.models.SimplexStep`:
    ``B_orig``, ``is_max_problem_original``.

Эти тесты проверяют **только структуру моделей**. Заполнение полей солвером
проверяется в `tests/test_canonical_vars.py` и acceptance-тестах.
"""

from dataclasses import fields
from fractions import Fraction

import pytest

from core.models import CanonicalVar, SimplexStep


# =============================================================================
# CanonicalVar — базовая структура.
# =============================================================================

class TestCanonicalVar:
    def test_create_orig(self):
        cv = CanonicalVar(kind='orig', ext_index=0, display_label='x_{1}', orig_index=0)
        assert cv.kind == 'orig'
        assert cv.ext_index == 0
        assert cv.display_label == 'x_{1}'
        assert cv.orig_index == 0
        assert cv.constraint_row is None

    def test_create_slack(self):
        cv = CanonicalVar(
            kind='slack', ext_index=3, display_label='s_{1}', constraint_row=0,
        )
        assert cv.kind == 'slack'
        assert cv.constraint_row == 0
        assert cv.orig_index is None

    def test_create_surplus(self):
        cv = CanonicalVar(
            kind='surplus', ext_index=4, display_label='s_{2}', constraint_row=1,
        )
        assert cv.kind == 'surplus'

    def test_create_artificial(self):
        cv = CanonicalVar(
            kind='artificial', ext_index=5, display_label='a_{1}', constraint_row=1,
        )
        assert cv.kind == 'artificial'

    def test_create_split_plus_minus(self):
        cv_plus = CanonicalVar(
            kind='split+', ext_index=0, display_label='x_{1}^{+}', orig_index=0,
        )
        cv_minus = CanonicalVar(
            kind='split-', ext_index=1, display_label='x_{1}^{-}', orig_index=0,
        )
        assert cv_plus.kind == 'split+'
        assert cv_minus.kind == 'split-'
        assert cv_plus.orig_index == cv_minus.orig_index == 0


# =============================================================================
# SimplexStep — новые опциональные поля.
# =============================================================================

class TestSimplexStepNewFields:
    def _make_minimal(self) -> SimplexStep:
        """Минимальный шаг без новых полей: должен создаваться без аргументов."""
        return SimplexStep(
            iteration=1,
            N=[0],
            B_inv=[[Fraction(1)]],
            x_B=[Fraction(0)],
            u_0=[Fraction(0)],
        )

    def test_B_orig_field_exists_and_defaults_to_none(self):
        step = self._make_minimal()
        assert hasattr(step, 'B_orig')
        assert step.B_orig is None

    def test_is_max_problem_original_field_exists_and_defaults_to_none(self):
        step = self._make_minimal()
        assert hasattr(step, 'is_max_problem_original')
        assert step.is_max_problem_original is None

    def test_B_orig_can_be_assigned(self):
        step = self._make_minimal()
        step.B_orig = [[Fraction(1), Fraction(2)], [Fraction(3), Fraction(4)]]
        assert step.B_orig[1][0] == Fraction(3)

    def test_is_max_problem_original_can_be_true_or_false(self):
        s1 = self._make_minimal()
        s1.is_max_problem_original = True
        s2 = self._make_minimal()
        s2.is_max_problem_original = False
        assert s1.is_max_problem_original is True
        assert s2.is_max_problem_original is False

    def test_old_fields_not_broken(self):
        """Backwards-compat: остальные поля SimplexStep — без изменений."""
        names = {f.name for f in fields(SimplexStep)}
        for required in (
            'iteration', 'N', 'B_inv', 'x_B', 'u_0', 'x_full', 'phase',
            'is_optimal', 'is_unbounded', 'is_infeasible',
            'j_0', 'z_0', 't_0', 's_0', 'description', 'c_B', 'diffs', 'ratios',
            'artificial_indices', 'var_map', 'u_0_original', 'row_inverted',
            'validation_errors',
        ):
            assert required in names, f"поле {required!r} удалено из SimplexStep"
