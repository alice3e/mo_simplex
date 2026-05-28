"""Тесты для корректного формирования блока двойственной задачи (D1 + D2).

Покрывает дефекты из FIX_PLAN.md:
    D1 — знаки переменных u_i в зависимости от типа i-го прямого ограничения
         (и направления оптимизации);
    D2 — знаки j-х двойственных ограничений в зависимости от типа x_j
         (свободная → '=', стандартная → '≤'/'≥' по is_max).
"""

from fractions import Fraction

from core.exporter import Exporter
from core.models import LinearProblem


def _dual_lines(c, A, b, signs, is_max=True, lb=None, ub=None) -> str:
    """Утилита: рендерит блок двойственной задачи в виде единой строки."""
    p = LinearProblem(
        c=[Fraction(x) for x in c],
        A=[[Fraction(x) for x in r] for r in A],
        b=[Fraction(x) for x in b],
        signs=signs,
        is_max=is_max,
        lower_bounds=(
            [Fraction(v) if v is not None else None for v in lb]
            if lb is not None else None
        ),
        upper_bounds=(
            [Fraction(v) if v is not None else None for v in ub]
            if ub is not None else None
        ),
    )
    return "\n".join(Exporter._format_dual_block(p))


# =============================================================================
# Утилитарные функции (D1, D2) — модульные проверки.
# =============================================================================

class TestDualVarSignUtility:
    """Проверка [`_dual_var_sign`](core/exporter.py)."""

    def test_min_equality_constraint_makes_u_free(self):
        assert Exporter._dual_var_sign('=', is_max=False) == '\\in \\mathbb{R}'

    def test_min_geq_constraint_makes_u_nonneg(self):
        assert Exporter._dual_var_sign('>=', is_max=False) == '\\geq 0'

    def test_min_leq_constraint_makes_u_nonpos(self):
        assert Exporter._dual_var_sign('<=', is_max=False) == '\\leq 0'

    def test_max_equality_constraint_makes_u_free(self):
        assert Exporter._dual_var_sign('=', is_max=True) == '\\in \\mathbb{R}'

    def test_max_leq_constraint_makes_u_nonneg(self):
        assert Exporter._dual_var_sign('<=', is_max=True) == '\\geq 0'

    def test_max_geq_constraint_makes_u_nonpos(self):
        assert Exporter._dual_var_sign('>=', is_max=True) == '\\leq 0'


class TestDualConstrSignUtility:
    """Проверка [`_dual_constr_sign`](core/exporter.py)."""

    @staticmethod
    def _make(is_max=True, lb=None, ub=None):
        return LinearProblem(
            c=[Fraction(1), Fraction(1)],
            A=[[Fraction(1), Fraction(1)]],
            b=[Fraction(1)],
            signs=['<='],
            is_max=is_max,
            lower_bounds=(
                [Fraction(v) if v is not None else None for v in lb]
                if lb is not None else None
            ),
            upper_bounds=(
                [Fraction(v) if v is not None else None for v in ub]
                if ub is not None else None
            ),
        )

    def test_free_variable_gives_equality(self):
        p = self._make(is_max=True, lb=[None, 0], ub=[None, None])
        assert Exporter._dual_constr_sign(0, p) == '='

    def test_standard_max_gives_geq(self):
        p = self._make(is_max=True)
        assert Exporter._dual_constr_sign(0, p) == '\\geq'
        assert Exporter._dual_constr_sign(1, p) == '\\geq'

    def test_standard_min_gives_leq(self):
        p = self._make(is_max=False)
        assert Exporter._dual_constr_sign(0, p) == '\\leq'

    def test_free_variable_same_for_min_and_max(self):
        p_max = self._make(is_max=True, lb=[None, 0], ub=[None, None])
        p_min = self._make(is_max=False, lb=[None, 0], ub=[None, None])
        assert Exporter._dual_constr_sign(0, p_max) == '='
        assert Exporter._dual_constr_sign(0, p_min) == '='


# =============================================================================
# Интеграционные проверки полного блока (с шаблоном).
# =============================================================================

class TestDualBlockMinMixed:
    """min, signs=['=','>=','<='] (эталонная задача профессора)."""

    def test_dual_vars_have_per_constraint_signs(self):
        dual = _dual_lines(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )
        assert "u_{1} &\\in \\mathbb{R}" in dual
        assert "u_{2} &\\geq 0" in dual
        assert "u_{3} &\\leq 0" in dual

    def test_dual_block_does_not_contain_old_uniform_u_geq_0(self):
        dual = _dual_lines(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )
        # Старая обобщённая строка должна полностью исчезнуть из вывода.
        assert "u &\\geq 0" not in dual
        assert "u &\\leq 0" not in dual

    def test_dual_constraints_use_leq_for_min_with_standard_vars(self):
        dual = _dual_lines(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )
        # Все x_j ≥ 0 → знак j-х двойственных ограничений: ≤
        assert "&\\leq 46" in dual
        assert "&\\leq 10" in dual
        assert "&\\leq 14" in dual


class TestDualBlockMaxAllLeq:
    """max, signs=['<=','<=','<='] — классика, все u_i ≥ 0."""

    def test_all_u_geq_0(self):
        dual = _dual_lines(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        for i in (1, 2, 3):
            assert f"u_{{{i}}} &\\geq 0" in dual

    def test_dual_constraints_use_geq_for_max_with_standard_vars(self):
        dual = _dual_lines(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        assert "&\\geq 12" in dual
        assert "&\\geq 3" in dual


class TestDualBlockMaxMixedSigns:
    """max, signs=['<=','=','>='] — должен выдать u_1≥0, u_2 свободна, u_3≤0."""

    def test_dual_var_signs(self):
        dual = _dual_lines(
            c=[4, 1],
            A=[[1, 2], [1, 1], [1, 0]],
            b=[6, 3, 1],
            signs=['<=', '=', '>='],
            is_max=True,
        )
        assert "u_{1} &\\geq 0" in dual
        assert "u_{2} &\\in \\mathbb{R}" in dual
        assert "u_{3} &\\leq 0" in dual


class TestDualBlockMinAllGeq:
    """min, signs=['>=','>=']."""

    def test_all_u_geq_0(self):
        dual = _dual_lines(
            c=[2, 3],
            A=[[1, 1], [2, 1]],
            b=[3, 4],
            signs=['>=', '>='],
            is_max=False,
        )
        assert "u_{1} &\\geq 0" in dual
        assert "u_{2} &\\geq 0" in dual


class TestDualBlockFreeVariable:
    """max c=(1,1), x_1 свободная → первое двойственное ограничение — равенство."""

    def test_first_constraint_is_equality(self):
        dual = _dual_lines(
            c=[1, 1],
            A=[[1, 1]],
            b=[10],
            signs=['<='],
            is_max=True,
            lb=[None, 0],
            ub=[None, None],
        )
        # Первое (по x_1) — равенство:
        assert ("&= 1" in dual) or ("& = 1" in dual)
        # Второе (по x_2, стандартная, max) — `≥ 1`:
        assert "&\\geq 1" in dual
