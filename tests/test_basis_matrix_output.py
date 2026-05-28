"""Тесты вывода исходной базисной матрицы B_orig и пометки о порядке строк (Этап 5).

Покрывает дефекты:
    D4 — каждый шаг содержит ИСХОДНУЮ матрицу B_orig
         перед обратной B (нотация учебника `B = ...` сохранена).
    D6 — под обратной B стоит пояснение о порядке строк (соответствие
         i-му ИСХОДНОМУ ограничению, не i-й базисной переменной).
"""

from fractions import Fraction

import pytest

from core.exporter import Exporter
from core.models import LinearProblem
from core.solver import SimplexSolver


def _run(c, A, b, signs, is_max=True):
    p = LinearProblem(
        c=[Fraction(x) for x in c],
        A=[[Fraction(x) for x in r] for r in A],
        b=[Fraction(x) for x in b],
        signs=signs,
        is_max=is_max,
    )
    s = SimplexSolver(p)
    return p, s, list(s.solve())


# =============================================================================
# D4: исходная базисная матрица.
# =============================================================================

class TestBOrigOutput:
    @pytest.fixture
    def setup(self):
        return _run(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_step_contains_b_orig_label(self, setup):
        _, _, steps = setup
        rendered = "\n".join(
            Exporter._render_step_lines(steps[-1], False, Fraction(43000, 7))
        )
        assert "Исходная базисная матрица" in rendered
        assert "B_{\\text{orig}}" in rendered

    def test_b_orig_precedes_b_inverse(self, setup):
        _, _, steps = setup
        rendered = "\n".join(
            Exporter._render_step_lines(steps[-1], False, Fraction(43000, 7))
        )
        i_orig = rendered.index("Исходная базисная матрица")
        i_inv = rendered.index("Обратная базисная матрица")
        assert i_orig < i_inv, (
            "исходная матрица должна выводиться ПЕРЕД обратной"
        )

    def test_b_orig_columns_match_basis(self, setup):
        """Содержимое B_orig — это столбцы A по индексам из N в порядке N."""
        _, s, steps = setup
        last = steps[-1]
        assert last.B_orig is not None
        m = s.n_constraints
        for k, idx in enumerate(last.N):
            for i in range(m):
                assert last.B_orig[i][k] == s.A[i][idx]

    def test_b_orig_present_in_every_step(self, setup):
        """B_orig заполнено в КАЖДОМ шаге, не только финальном."""
        _, _, steps = setup
        for step in steps:
            assert step.B_orig is not None, (
                f"шаг {step.iteration} ({step.phase=}) без B_orig"
            )


# =============================================================================
# D6: пометка о порядке строк B.
# =============================================================================

class TestBasisRowOrderNote:
    @pytest.fixture
    def setup(self):
        return _run(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_note_present(self, setup):
        _, _, steps = setup
        rendered = "\n".join(
            Exporter._render_step_lines(steps[-1], False, Fraction(43000, 7))
        )
        assert "соответствует $i$-му исходному" in rendered

    def test_note_lists_all_basis_vars(self, setup):
        _, _, steps = setup
        last = steps[-1]
        rendered = "\n".join(
            Exporter._render_step_lines(last, False, Fraction(43000, 7))
        )
        # Должны упоминаться все строки 1..m
        for i in range(1, len(last.N) + 1):
            assert f"строка {i} →" in rendered


# =============================================================================
# Регрессия: B_orig для классической задачи max (только <=).
# =============================================================================

class TestBOrigForPureMaxLeq:
    def test_b_orig_for_max_classic(self):
        _, s, steps = _run(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        last = steps[-1]
        # Стартовый базис — все balancing-переменные (единичная матрица),
        # затем после поворотов могут быть исходные A-столбцы.
        # Главное: B_orig определён и совпадает со столбцами A по N.
        assert last.B_orig is not None
        for k, idx in enumerate(last.N):
            for i in range(s.n_constraints):
                assert last.B_orig[i][k] == s.A[i][idx]
