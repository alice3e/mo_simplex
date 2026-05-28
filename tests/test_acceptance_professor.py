"""Acceptance (end-to-end) тесты для исправлений из FIX_PLAN.md.

Эти тесты проверяют ИТОГОВОЕ поведение всего pipeline (solve → HTML/Markdown)
и должны падать ДО исправлений из FIX_PLAN.md (этапы 1-7) и проходить ПОСЛЕ.

Покрываемые дефекты:
    D1 — знаки u_i в двойственной задаче по типам ограничений прямой;
    D2 — знаки двойственных ограничений (для свободных x_j — равенство);
    D3 — каноническая форма (отдельный блок с s_i, a_i и таблицей соответствия);
    D4 — исходная базисная матрица B_orig (столбцы A_j для j∈N);
    D5 — на финальном шаге min-задачи основной u^* = u_0_original;
    D6 — пометка о соответствии строк B исходным ограничениям;
        + использование меток s_i/a_i в подписи базиса каждого шага.

Эталонная задача (как у профессора):

    min 46x1 + 10x2 + 14x3
    s.t.   x1 +  x2 +  x3 = 500
         81x1 + 2x2 + 16x3 >= 5000
          5x1 +22x2 +  6x3 <= 7000
    x1, x2, x3 >= 0

    Ответ: f* = 43000/7, x* = (0, 1500/7, 2000/7),
           u* = (66/7, 2/7, 0) — в исходных терминах (y_1 ∈ R, y_2 ≥ 0, y_3 ≤ 0).
"""

from fractions import Fraction

import pytest

from core.exporter import Exporter
from core.models import LinearProblem
from core.solver import SimplexSolver


# =============================================================================
# Хелперы
# =============================================================================

def _build_solver(c, A, b, signs, is_max=False, lower_bounds=None, upper_bounds=None):
    """Конструирует LinearProblem + SimplexSolver, прогоняет solve()."""
    p = LinearProblem(
        c=[Fraction(x) for x in c],
        A=[[Fraction(x) for x in r] for r in A],
        b=[Fraction(x) for x in b],
        signs=signs,
        is_max=is_max,
        lower_bounds=(
            [Fraction(v) if v is not None else None for v in lower_bounds]
            if lower_bounds is not None else None
        ),
        upper_bounds=(
            [Fraction(v) if v is not None else None for v in upper_bounds]
            if upper_bounds is not None else None
        ),
    )
    s = SimplexSolver(p)
    steps = list(s.solve())
    return p, s, steps


def _render_full_html(problem, solver, steps):
    """Рендерит ИТОГОВЫЙ HTML — тот, что увидит пользователь через UI/Api.solve().

    Прокидывает все метаданные канонической формы, чтобы соответствовать
    тому, как :meth:`main.Api.solve` вызывает Exporter.
    """
    last = steps[-1]
    x_original = None
    final_answer = None
    if last.is_optimal and last.phase == 2:
        x_original = solver.recover_original_x(last.x_full)
        final_answer = solver.compute_final_answer(last)
    return Exporter.generate_html(
        problem, steps, final_answer,
        detailed=False,
        x_original=x_original,
        n_orig_vars=solver.n_orig_vars,
        canonical_vars=solver.canonical_vars,
        canonical_A=solver.A,
        canonical_b=solver.b,
        canonical_signs_after_norm=solver.signs,
        artificial_indices=list(solver.artificial),
    )


# =============================================================================
# Кейс 1. Эталонная задача профессора.
# =============================================================================

class TestProfessorReference:
    """Точная задача из эталонного разбора: min, =/≥/≤, 3 переменные.

    Все 6 acceptance-проверок концентрированы здесь.
    """

    @pytest.fixture
    def setup(self):
        p, s, steps = _build_solver(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )
        return p, s, steps

    # -------- baseline: что в ядре уже работает (sanity check) ---------------
    def test_baseline_optimal_value(self, setup):
        _, s, steps = setup
        last = steps[-1]
        assert last.is_optimal and last.phase == 2
        assert s.compute_final_answer(last) == Fraction(43000, 7)

    def test_baseline_optimal_x(self, setup):
        _, s, steps = setup
        last = steps[-1]
        x = s.recover_original_x(last.x_full)
        assert x == [Fraction(0), Fraction(1500, 7), Fraction(2000, 7)]

    def test_baseline_dual_u_original(self, setup):
        _, s, steps = setup
        last = steps[-1]
        # ВНУТРЕННИЕ u_0 — для max-формы (с минусами):
        assert last.u_0 == [Fraction(-66, 7), Fraction(-2, 7), Fraction(0)]
        # u_0_original — то, что должно ПОКАЗЫВАТЬСЯ пользователю:
        assert last.u_0_original == [Fraction(66, 7), Fraction(2, 7), Fraction(0)]

    # -------- D1: знаки переменных u_i в двойственной задаче ----------------
    def test_d1_dual_var_signs(self, setup):
        """signs=['=','>=','<='], is_max=False:
              u_1 ∈ R,  u_2 ≥ 0,  u_3 ≤ 0.

        В текущем коде вместо этого выводится ОДНА строка `u ≥ 0`.
        """
        p, _, _ = setup
        dual = "\n".join(Exporter._format_dual_block(p))

        # Новое поведение: построчные знаки переменных.
        assert "u_{1} &\\in \\mathbb{R}" in dual, (
            "u_1 должна быть свободной (соответствует ограничению-равенству)"
        )
        assert "u_{2} &\\geq 0" in dual, (
            "u_2 ≥ 0 (соответствует ≥-ограничению для min)"
        )
        assert "u_{3} &\\leq 0" in dual, (
            "u_3 ≤ 0 (соответствует ≤-ограничению для min)"
        )
        # Старая общая строка должна исчезнуть:
        assert "u &\\geq 0" not in dual, (
            "обобщённая строка 'u ≥ 0' больше не используется"
        )

    # -------- D3: каноническая форма с метками s_i, a_i ----------------------
    def test_d3_canonical_form_block_present(self, setup):
        """В шапке HTML должен быть блок «Каноническая форма» с метками s_i и a_i."""
        p, s, steps = setup
        html = _render_full_html(p, s, steps)

        assert "Каноническая форма" in html, (
            "ожидаем явный заголовок блока «Каноническая форма»"
        )
        # Метки балансовой/избыточной/искусственных переменных:
        assert "s_{3}" in html or "s_3" in html, (
            "должна присутствовать балансовая переменная s_3 (для <=)"
        )
        assert "s_{2}" in html or "s_2" in html, (
            "должна присутствовать избыточная переменная s_2 (для >=)"
        )
        assert "a_{1}" in html or "a_1" in html, (
            "должна присутствовать искусственная a_1 (для =)"
        )
        assert "a_{2}" in html or "a_2" in html, (
            "должна присутствовать искусственная a_2 (для >=)"
        )
        # Целевая фазы I:
        assert "w" in html and ("\\to \\min" in html or "→ min" in html)

    def test_d3_step_basis_uses_named_labels(self, setup):
        """В подписи базиса каждого шага должны быть s_i/a_i, не безликие x_k."""
        p, s, steps = setup
        html = _render_full_html(p, s, steps)
        # В финальном базисе [2, 6, 1] = (x_3, s_3, x_2) — должно быть видно s_3, не x_7.
        # Сейчас в текущем коде там стоит "x_7", который вводит в заблуждение.
        assert "s_{3}" in html or "s_3" in html, (
            "финальный базис должен содержать метку s_3, а не безликую x_7"
        )

    # -------- D4: исходная базисная матрица B_orig --------------------------
    def test_d4_B_orig_present_in_step(self, setup):
        """Каждый шаг должен содержать ИСХОДНУЮ базисную матрицу (B_orig)
        перед обратной (B). Подпись `Исходная базисная матрица`."""
        p, s, steps = setup
        html = _render_full_html(p, s, steps)

        assert "Исходная базисная матрица" in html, (
            "должна выводиться исходная B_orig (как у профессора в эталоне)"
        )

    def test_d4_B_orig_columns_match_basis(self, setup):
        """Содержимое B_orig для финального шага должно совпадать со столбцами
        исходной A с индексами из N в порядке N."""
        _, s, steps = setup
        last = steps[-1]
        # После реализации D4 SimplexStep получит поле B_orig.
        assert getattr(last, 'B_orig', None) is not None, (
            "SimplexStep.B_orig должен быть заполнен солвером"
        )
        m = s.n_constraints
        for k, idx in enumerate(last.N):
            for i in range(m):
                assert last.B_orig[i][k] == s.A[i][idx], (
                    f"B_orig[{i}][{k}] != A[{i}][{idx}]"
                )

    # -------- D5: основной u^* на финале — u_0_original ---------------------
    def test_d5_final_u_star_is_u_original(self, setup):
        """На финальном шаге блок «Двойственное решение u^*» должен использовать
        ПОЛОЖИТЕЛЬНЫЕ значения (66/7, 2/7, 0), а не внутренние (-66/7, -2/7, 0)."""
        p, s, steps = setup
        html = _render_full_html(p, s, steps)

        # Положительные знаки в формуле u^*:
        assert "\\dfrac{66}{7}" in html, (
            "u^* должен содержать +66/7 как ОСНОВНОЕ значение"
        )
        assert "\\dfrac{2}{7}" in html, (
            "u^* должен содержать +2/7"
        )
        # Текущее поведение: u^* = u_0 = [-66/7, -2/7, 0]
        # ПОСЛЕ исправления это значение либо не показывается на финале,
        # либо вынесено в свёрнутый блок.
        # Считаем количество вхождений "u^*": ожидаем, что в первой формуле
        # стоит ПОЛОЖИТЕЛЬНЫЙ вектор.

        # Жёсткое требование: формула вида `u^* = u_0 = [-66/7, ...]` исчезла:
        assert "u^* = u_0 = \\begin{bmatrix} -\\dfrac{66}{7}" not in html, (
            "основная формула u^* не должна использовать внутренний u_0 "
            "с отрицательными знаками для min-задачи"
        )

    # -------- D6: пометка о порядке строк B ---------------------------------
    def test_d6_basis_row_note_present(self, setup):
        """Под обратной базисной матрицей должна быть пояснительная строка
        о соответствии строк ИСХОДНЫМ ограничениям, а не базисным переменным."""
        p, s, steps = setup
        html = _render_full_html(p, s, steps)

        # Любая из формулировок: «строка i соответствует i-му ограничению»
        marker_variants = [
            "соответствует $i$-му ИСХОДНОМУ ограничению",
            "соответствует $i$-му исходному ограничению",
            "соответствует i-му ограничению",
        ]
        assert any(m in html for m in marker_variants), (
            "должна присутствовать пояснительная пометка о порядке строк B"
        )


# =============================================================================
# Кейс 2. min с двумя ≥ — D1: u_1, u_2 ≥ 0; нет искусственной с = или ≤.
# =============================================================================

class TestMinTwoGeq:
    """min 2x1 + 3x2,  x1+x2 ≥ 3,  2x1+x2 ≥ 4.
    Ожидаем: u_1 ≥ 0, u_2 ≥ 0; каноническая форма содержит s_1, s_2 (избыточные)
    и a_1, a_2 (искусственные), но не балансовые."""

    @pytest.fixture
    def setup(self):
        return _build_solver(
            c=[2, 3],
            A=[[1, 1], [2, 1]],
            b=[3, 4],
            signs=['>=', '>='],
            is_max=False,
        )

    def test_d1_both_geq_min_means_u_geq_0(self, setup):
        p, _, _ = setup
        dual = "\n".join(Exporter._format_dual_block(p))
        assert "u_{1} &\\geq 0" in dual
        assert "u_{2} &\\geq 0" in dual
        # Не должно остаться обобщённой строки:
        assert "u &\\geq 0" not in dual

    def test_d3_canonical_only_surplus_and_artificials(self, setup):
        p, s, steps = setup
        html = _render_full_html(p, s, steps)
        assert "Каноническая форма" in html
        # Две избыточные s_1, s_2:
        for name in ("s_1", "s_{1}"):
            if name in html:
                break
        else:
            pytest.fail("должна присутствовать избыточная s_1")
        # Две искусственные:
        assert ("a_1" in html) or ("a_{1}" in html)
        assert ("a_2" in html) or ("a_{2}" in html)


# =============================================================================
# Кейс 3. max со смешанными ограничениями — D1: u_1 ≥ 0, u_2 ∈ R, u_3 ≤ 0.
# =============================================================================

class TestMaxMixedSigns:
    """max 4x1 + x2,  x1+2x2 ≤ 6,  x1+x2 = 3,  x1 ≥ 1 (фиктивное ≥).
    Для max:  ≤ → u≥0;  = → u свободна;  ≥ → u≤0."""

    @pytest.fixture
    def setup(self):
        return _build_solver(
            c=[4, 1],
            A=[[1, 2], [1, 1], [1, 0]],
            b=[6, 3, 1],
            signs=['<=', '=', '>='],
            is_max=True,
        )

    def test_d1_dual_var_signs_for_max(self, setup):
        p, _, _ = setup
        dual = "\n".join(Exporter._format_dual_block(p))
        assert "u_{1} &\\geq 0" in dual, "≤ для max ⇒ u_1 ≥ 0"
        assert "u_{2} &\\in \\mathbb{R}" in dual, "= ⇒ u_2 свободна"
        assert "u_{3} &\\leq 0" in dual, "≥ для max ⇒ u_3 ≤ 0"
        assert "u &\\geq 0" not in dual


# =============================================================================
# Кейс 4. D2 — свободная переменная даёт двойственное РАВЕНСТВО.
# =============================================================================

class TestDualConstraintForFreeVar:
    """max x1 + x2,  x1+x2 ≤ 10,  x1 свободная (lb=None, ub=None).
    Для свободной x_1 двойственное ограничение по столбцу 1 — РАВЕНСТВО:
        u_1 = 1
    а для x_2 (стандартная, max) — u_1 ≥ 1.
    """

    @pytest.fixture
    def setup(self):
        return _build_solver(
            c=[1, 1],
            A=[[1, 1]],
            b=[10],
            signs=['<='],
            is_max=True,
            lower_bounds=[None, 0],
            upper_bounds=[None, None],
        )

    def test_d2_free_var_gives_equality_constraint(self, setup):
        p, _, _ = setup
        dual = "\n".join(Exporter._format_dual_block(p))
        # В первом ограничении (для x_1, свободной) знак — равенство.
        # Берём строку, содержащую "&= 1" или "& = 1" (т.е. = 1).
        assert ("&= 1" in dual) or ("& = 1" in dual), (
            "для свободной x_1 первое двойственное ограничение должно быть равенством"
        )
        # Для x_2 (стандартная ≥0, max) — должно быть `≥ 1`.
        assert ("&\\geq 1" in dual) or ("& \\geq 1" in dual), (
            "для стандартной x_2 двойственное ограничение должно быть ≥ 1"
        )
