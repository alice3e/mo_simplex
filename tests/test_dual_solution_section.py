"""Тесты блока «Двойственное решение u^*» на финальном шаге (Этап 6).

Покрывает D5: для min-задачи как ОСНОВНОЙ должен выводиться `u^* = u_0_original`
(правильные знаки исходной задачи), а внутренний `u_0` для приведённой
max-формы — только справочный блок.
"""

from fractions import Fraction

import pytest

from core.exporter import Exporter
from core.models import LinearProblem, SimplexStep
from core.solver import SimplexSolver


def _run(c, A, b, signs, is_max):
    p = LinearProblem(
        c=[Fraction(x) for x in c],
        A=[[Fraction(x) for x in r] for r in A],
        b=[Fraction(x) for x in b],
        signs=signs,
        is_max=is_max,
    )
    s = SimplexSolver(p)
    return p, s, list(s.solve())


def _render_final(steps, solver, p):
    last = steps[-1]
    final_answer = solver.compute_final_answer(last) if last.is_optimal and last.phase == 2 else None
    return "\n".join(Exporter._render_step_lines(
        last, False, final_answer,
        x_original=solver.recover_original_x(last.x_full) if last.is_optimal and last.phase == 2 else None,
        n_orig_vars=solver.n_orig_vars,
    ))


# =============================================================================
# Эталонная задача профессора (min, =/≥/≤).
# =============================================================================

class TestProfessorMinFinalDual:
    @pytest.fixture
    def setup(self):
        return _run(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_u_star_uses_positive_values(self, setup):
        p, s, steps = setup
        rendered = _render_final(steps, s, p)
        # Ожидаем 66/7, 2/7 в формуле u^*:
        assert "\\dfrac{66}{7}" in rendered
        assert "\\dfrac{2}{7}" in rendered

    def test_main_u_star_formula_not_u_0(self, setup):
        """Основная формула u^* не должна быть u^* = u_0 = [-66/7,...]."""
        p, s, steps = setup
        rendered = _render_final(steps, s, p)
        # Старая формула, которую раньше выдавал Exporter:
        assert "u^* = u_0 = \\begin{bmatrix} -\\dfrac{66}{7}" not in rendered

    def test_min_disclaimer_present(self, setup):
        p, s, steps = setup
        rendered = _render_final(steps, s, p)
        assert "$\\min$" in rendered
        assert "противоположен" in rendered

    def test_internal_u_0_in_details_block(self, setup):
        """Внутренний u_0 (с минусами) попал в служебный <details>-блок."""
        p, s, steps = setup
        rendered = _render_final(steps, s, p)
        assert "<details>" in rendered
        # И именно внутри details упоминается приведённая max-форма:
        assert "приведённой $\\max$-формы" in rendered

    def test_strong_duality_check_uses_x_star_and_u_star(self, setup):
        p, s, steps = setup
        rendered = _render_final(steps, s, p)
        # Новая формула: f^* = c^T x^* = b^T u^*
        assert "f^* = c^T x^* = b^T u^*" in rendered


# =============================================================================
# max-задача без инверсий: u_0 == u_0_original → details-блока быть не должно.
# =============================================================================

class TestMaxNoInversion:
    def test_no_details_block_for_simple_max(self):
        p, s, steps = _run(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        rendered = _render_final(steps, s, p)
        # u_0 == u_0_original для этой задачи → служебный блок не нужен.
        assert "<details>" not in rendered

    def test_main_u_star_present(self):
        p, s, steps = _run(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        rendered = _render_final(steps, s, p)
        # u^* = [3, 0, 0] — основной вектор показан:
        assert "u^*" in rendered


# =============================================================================
# Backwards-compat: fallback при отсутствии u_0_original.
# =============================================================================

class TestFallbackWhenNoUOriginal:
    def test_fallback_to_u_0_when_no_original(self):
        """Если u_0_original=None — _render_dual_solution использует старое поведение."""
        step = SimplexStep(
            iteration=1,
            N=[0],
            B_inv=[[Fraction(1)]],
            x_B=[Fraction(0)],
            u_0=[Fraction(7)],
            phase=2,
            is_optimal=True,
            u_0_original=None,
        )
        rendered = "\n".join(Exporter._render_dual_solution(step))
        # Должна быть формула вида `u^* = u_0 = [7]`:
        assert "u^* = u_0 = \\begin{bmatrix} 7 \\end{bmatrix}" in rendered
