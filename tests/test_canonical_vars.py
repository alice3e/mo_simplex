"""Тесты заполнения `SimplexSolver.canonical_vars` (Этап 3).

Проверяют, что после `_to_canonical` солвер корректно регистрирует все
расширенные переменные с правильными:
    - типами (`kind`: orig / slack / surplus / artificial / split±),
    - индексами `ext_index` (совпадают с позициями столбцов в `self.A`),
    - метками `display_label` (s_i, a_k, x_j^+, x_j^-),
    - привязкой к строке ограничения (`constraint_row`) — для slack/surplus/artificial,
    - привязкой к исходной переменной (`orig_index`) — для orig/split±.
"""

from fractions import Fraction

import pytest

from core.models import CanonicalVar, LinearProblem
from core.solver import SimplexSolver


def _solver(c, A, b, signs, is_max=True, lb=None, ub=None, canonical_mode=False):
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
    return SimplexSolver(p, canonical_mode=canonical_mode)


# =============================================================================
# Эталонная задача профессора: 3 переменные, =/≥/≤ → orig×3 + s_2 + a_1, a_2 + s_3.
# =============================================================================

class TestProfessorReferenceCanonicalVars:
    @pytest.fixture
    def solver(self):
        return _solver(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_total_count(self, solver):
        # 3 orig + 1 a_1 (=) + 1 s_2 (surplus для ≥) + 1 a_2 (искусственная ≥) + 1 s_3 (≤)
        assert len(solver.canonical_vars) == 7
        assert solver.n_vars == 7

    def test_orig_vars_first_three(self, solver):
        labels = [cv.display_label for cv in solver.canonical_vars[:3]]
        assert labels == ['x_{1}', 'x_{2}', 'x_{3}']
        for k, cv in enumerate(solver.canonical_vars[:3]):
            assert cv.kind == 'orig'
            assert cv.ext_index == k
            assert cv.orig_index == k
            assert cv.constraint_row is None

    def test_artificial_for_equality(self, solver):
        cv = solver.canonical_vars[3]
        assert cv.kind == 'artificial'
        assert cv.ext_index == 3
        assert cv.display_label == 'a_{1}'
        assert cv.constraint_row == 0  # ограничение `=` — индекс 0

    def test_surplus_then_artificial_for_geq(self, solver):
        # Для строки ≥ порядок: surplus, потом artificial.
        cv_s = solver.canonical_vars[4]
        cv_a = solver.canonical_vars[5]
        assert cv_s.kind == 'surplus' and cv_s.display_label == 's_{2}'
        assert cv_s.constraint_row == 1
        assert cv_a.kind == 'artificial' and cv_a.display_label == 'a_{2}'
        assert cv_a.constraint_row == 1

    def test_slack_for_leq(self, solver):
        cv = solver.canonical_vars[6]
        assert cv.kind == 'slack'
        assert cv.display_label == 's_{3}'
        assert cv.constraint_row == 2
        assert cv.ext_index == 6

    def test_artificial_indices_match(self, solver):
        artificial_ext = [cv.ext_index for cv in solver.canonical_vars if cv.kind == 'artificial']
        assert artificial_ext == solver.artificial


# =============================================================================
# Чистая <=-задача: только slack, без artificial.
# =============================================================================

class TestPureLeqCanonicalVars:
    def test_only_slack_no_artificial(self):
        s = _solver(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        # 2 исходных + 3 балансовых = 5
        assert len(s.canonical_vars) == 5
        labels = [cv.display_label for cv in s.canonical_vars]
        assert labels == ['x_{1}', 'x_{2}', 's_{1}', 's_{2}', 's_{3}']

        # Нет искусственных:
        assert s.artificial == []
        kinds = [cv.kind for cv in s.canonical_vars]
        assert 'artificial' not in kinds


# =============================================================================
# Две >= — два surplus + два artificial.
# =============================================================================

class TestTwoGeqCanonicalVars:
    def test_surplus_and_artificial_per_geq_row(self):
        s = _solver(
            c=[2, 3],
            A=[[1, 1], [2, 1]],
            b=[3, 4],
            signs=['>=', '>='],
            is_max=False,
        )
        # 2 orig + 2*(s_i + a_i) = 6
        assert len(s.canonical_vars) == 6
        labels = [cv.display_label for cv in s.canonical_vars]
        assert labels == [
            'x_{1}', 'x_{2}',
            's_{1}', 'a_{1}',
            's_{2}', 'a_{2}',
        ]
        # Artificial должны быть строго [3, 5]
        assert s.artificial == [3, 5]


# =============================================================================
# Свободная переменная: split+ / split-.
# =============================================================================

class TestFreeVariableCanonicalVars:
    def test_split_for_free_variable(self):
        # max x1+x2, x1+x2<=10, x1 свободная, x2 стандартная.
        s = _solver(
            c=[1, 1],
            A=[[1, 1]],
            b=[10],
            signs=['<='],
            is_max=True,
            lb=[None, 0],
            ub=[None, None],
        )
        # x_1 → split+, split-; x_2 → orig; +1 slack.
        kinds = [cv.kind for cv in s.canonical_vars]
        labels = [cv.display_label for cv in s.canonical_vars]
        assert kinds == ['split+', 'split-', 'orig', 'slack']
        assert labels == ['x_{1}^{+}', 'x_{1}^{-}', 'x_{2}', 's_{1}']

        # Привязка к исходной переменной:
        assert s.canonical_vars[0].orig_index == 0
        assert s.canonical_vars[1].orig_index == 0
        assert s.canonical_vars[2].orig_index == 1


# =============================================================================
# Фиксированная (lb = ub) — ничего не добавляется, столбец удаляется.
# =============================================================================

class TestFixedVariableCanonicalVars:
    def test_fixed_var_not_in_canonical_vars(self):
        # max x1+x2, x2<=5, x1 фиксирована lb=ub=2.
        s = _solver(
            c=[1, 1],
            A=[[0, 1]],
            b=[5],
            signs=['<='],
            is_max=True,
            lb=[2, 0],
            ub=[2, None],
        )
        # x_1 удалена. Остаётся x_2 + 1 slack = 2.
        labels = [cv.display_label for cv in s.canonical_vars]
        assert labels == ['x_{2}', 's_{1}']


# =============================================================================
# Канонический режим: авто-детектор ортов, без искусственных.
# =============================================================================

class TestCanonicalModeNoArtificials:
    def test_no_extra_vars_when_orts_found(self):
        s = _solver(
            c=[12, 3, 0, 0, 0],
            A=[
                [4, 1, 1, 0, 0],
                [2, 2, 0, 1, 0],
                [6, 3, 0, 0, 1],
            ],
            b=[16, 22, 36],
            signs=['=', '=', '='],
            is_max=True,
            canonical_mode=True,
        )
        # Никаких добавленных переменных — все 5 исходных, орты найдены автоматически.
        labels = [cv.display_label for cv in s.canonical_vars]
        assert labels == ['x_{1}', 'x_{2}', 'x_{3}', 'x_{4}', 'x_{5}']
        assert s.artificial == []


# =============================================================================
# B_orig: shape and correspondence.
# =============================================================================

class TestBOrigField:
    def test_final_step_B_orig_is_filled(self):
        s = _solver(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )
        last = list(s.solve())[-1]
        assert last.B_orig is not None
        # Размер m × m, m = 3
        assert len(last.B_orig) == 3
        assert all(len(r) == 3 for r in last.B_orig)
        # k-й столбец B_orig совпадает с self.A[*][self.N[k]]
        for k, idx in enumerate(last.N):
            for i in range(3):
                assert last.B_orig[i][k] == s.A[i][idx]

    def test_is_max_problem_original_propagated(self):
        # min-задача → is_max=False
        s_min = _solver(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )
        for step in s_min.solve():
            assert step.is_max_problem_original is False
        # max-задача → True
        s_max = _solver(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        for step in s_max.solve():
            assert step.is_max_problem_original is True
