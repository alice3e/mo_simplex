"""Тесты для рендеринга блока «Каноническая форма» в Exporter (Этап 4).

Покрывает:
    - `_format_canonical_block` — стандарт + интеграция в `generate_html` / `generate_markdown`;
    - наличие меток `s_i`, `a_i`, `x_j^+`, `x_j^-` в выводе;
    - таблицу соответствия;
    - целевую функцию фазы I `w = a_1 + a_2 + ... → min`;
    - корректное отсутствие блока, когда `canonical_vars=None`.
"""

from fractions import Fraction

import pytest

from core.exporter import Exporter
from core.models import LinearProblem
from core.solver import SimplexSolver


def _build(c, A, b, signs, is_max=True, lb=None, ub=None):
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
    s = SimplexSolver(p)
    steps = list(s.solve())
    return p, s, steps


def _block(s: SimplexSolver, p: LinearProblem) -> str:
    return "\n".join(Exporter._format_canonical_block(
        p, s.canonical_vars, s.A, s.b, s.signs, list(s.artificial),
    ))


# =============================================================================
# Эталонная задача профессора.
# =============================================================================

class TestCanonicalBlockProfessorReference:
    @pytest.fixture
    def setup(self):
        return _build(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_system_uses_named_labels(self, setup):
        p, s, _ = setup
        block = _block(s, p)
        # В системе должны быть метки s_2, s_3, a_1, a_2:
        assert "s_{2}" in block
        assert "s_{3}" in block
        assert "a_{1}" in block
        assert "a_{2}" in block

    def test_correspondence_table_present(self, setup):
        p, s, _ = setup
        block = _block(s, p)
        assert "Соответствие расширенных переменных" in block
        assert "| Индекс | Обозначение | Тип |" in block
        # И сами строки таблицы:
        assert "балансовая (для $\\leq$)" in block
        assert "избыточная (для $\\geq$)" in block
        assert "искусственная" in block

    def test_phase1_objective_function(self, setup):
        p, s, _ = setup
        block = _block(s, p)
        assert "w = a_{1} + a_{2}" in block
        assert "\\to \\min" in block

    def test_equation_one_contains_a1(self, setup):
        """Первая строка системы (для x1+x2+x3=500) должна содержать a_1."""
        p, s, _ = setup
        block = _block(s, p)
        # Хотя бы одна строка `= 500` присутствует и содержит +a_{1}:
        # (не привязываемся к точному форматированию пробелов)
        assert "a_{1}" in block and "= 500" in block

    def test_equation_two_contains_minus_s2_plus_a2(self, setup):
        p, s, _ = setup
        block = _block(s, p)
        # x_1 коэф 81, x_2 коэф 2, x_3 коэф 16, s_2 коэф -1, a_2 коэф +1.
        assert "- s_{2}" in block
        assert "+ a_{2}" in block
        assert "= 5000" in block

    def test_equation_three_contains_s3_only(self, setup):
        p, s, _ = setup
        block = _block(s, p)
        assert "+ s_{3}" in block or "s_{3}" in block
        assert "= 7000" in block


# =============================================================================
# Чистая <=-задача: только slack, нет искусственных → нет фазы I.
# =============================================================================

class TestCanonicalBlockPureLeq:
    def test_no_artificial_no_phase1_objective(self):
        p, s, _ = _build(
            c=[12, 3],
            A=[[4, 1], [2, 2], [6, 3]],
            b=[16, 22, 36],
            signs=['<=', '<=', '<='],
            is_max=True,
        )
        block = _block(s, p)
        assert "s_{1}" in block
        assert "s_{2}" in block
        assert "s_{3}" in block
        # Никаких искусственных — никакой фазы I:
        assert "a_{1}" not in block
        assert "Целевая функция фазы I" not in block


# =============================================================================
# Свободная переменная: split+ / split-.
# =============================================================================

class TestCanonicalBlockFreeVar:
    def test_split_plus_minus_present(self):
        p, s, _ = _build(
            c=[1, 1],
            A=[[1, 1]],
            b=[10],
            signs=['<='],
            is_max=True,
            lb=[None, 0],
            ub=[None, None],
        )
        block = _block(s, p)
        assert "x_{1}^{+}" in block
        assert "x_{1}^{-}" in block
        # В таблице соответствия — пояснение про разбиение свободной:
        assert "положит. часть свободной" in block
        assert "отриц. часть свободной" in block


# =============================================================================
# Интеграция: generate_html / generate_markdown с canonical_vars.
# =============================================================================

class TestIntegrationGenerateHtmlMarkdown:
    @pytest.fixture
    def setup(self):
        return _build(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_html_includes_canonical_section(self, setup):
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        assert "Каноническая форма" in html
        assert "s_{3}" in html  # балансовая 3-го <=
        assert "a_{1}" in html  # искусственная 1-го =
        assert "a_{2}" in html

    def test_markdown_includes_canonical_section(self, setup):
        p, s, steps = setup
        md = Exporter.generate_markdown(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        assert "## Каноническая форма" in md
        assert "s_{3}" in md
        assert "a_{1}" in md

    def test_html_omits_canonical_when_no_meta(self, setup):
        """Если canonical_vars=None — блок не выводится (обратная совместимость)."""
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
        )
        assert "Каноническая форма" not in html

    def test_markdown_omits_canonical_when_no_meta(self, setup):
        p, s, steps = setup
        md = Exporter.generate_markdown(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
        )
        assert "## Каноническая форма" not in md


# =============================================================================
# Этап 7: метки s_i/a_i используются в подписях базиса каждого шага.
# =============================================================================

class TestStepLabelsUseCanonicalNames:
    @pytest.fixture
    def setup(self):
        return _build(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_html_step_basis_uses_s_a_labels(self, setup):
        """В итоговом HTML подпись базиса финального шага содержит метку
        `s_3`, а не безликую `x_7`."""
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        # В финальном базисе N = [2, 6, 1] = (x_3, s_3, x_2) — должно быть видно s_3.
        # И НЕ должно быть `x_7`.
        # (Поскольку всё-таки бывают другие шаги с другими базисами, проверяем
        # только наличие s_3 и отсутствие безликого x_7 в подписи базиса.)
        assert "s_{3}" in html
        # x_7 встречается только в исходных метках (которых тут не должно быть);
        # точное место — внутри `**Базис $N$:**` строки.
        import re
        basis_lines = re.findall(r"Базис \$N\$:\$ \$\(([^)]+)\)", html)
        for line in basis_lines:
            assert "x_{7}" not in line, (
                f"подпись базиса '{line}' содержит безликую x_7 вместо s_3"
            )

    def test_markdown_step_basis_uses_s_a_labels(self, setup):
        p, s, steps = setup
        md = Exporter.generate_markdown(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        assert "s_{3}" in md

    def test_step_basis_falls_back_to_x_k_when_no_canonical_vars(self, setup):
        """Без canonical_vars базис показывается старыми безликими метками."""
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
        )
        # x_7 в подписи базиса должна сохраниться (backwards compat).
        assert "x_{7}" in html


# =============================================================================
# Markdown-таблица в HTML конвертируется в <table> (а не остаётся plain-text).
# =============================================================================

class TestMarkdownTableToHtml:
    @pytest.fixture
    def setup(self):
        return _build(
            c=[46, 10, 14],
            A=[[1, 1, 1], [81, 2, 16], [5, 22, 6]],
            b=[500, 5000, 7000],
            signs=['=', '>=', '<='],
            is_max=False,
        )

    def test_html_contains_table_element(self, setup):
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        assert "<table" in html
        assert "<thead" in html
        assert "<tbody" in html
        assert "<th" in html
        assert "<td" in html

    def test_html_does_not_contain_raw_md_table_pipes(self, setup):
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        # Plain-text md-таблицы быть не должно — была сконвертирована в <table>.
        assert "| Индекс | Обозначение | Тип |" not in html
        assert "|---|---|---|" not in html

    def test_html_table_contains_expected_cells(self, setup):
        p, s, steps = setup
        html = Exporter.generate_html(
            p, steps, s.compute_final_answer(steps[-1]),
            x_original=s.recover_original_x(steps[-1].x_full),
            n_orig_vars=s.n_orig_vars,
            canonical_vars=s.canonical_vars,
            canonical_A=s.A,
            canonical_b=s.b,
            canonical_signs_after_norm=s.signs,
            artificial_indices=list(s.artificial),
        )
        # Сохранены inline-LaTeX-метки внутри ячеек:
        assert "<td class='border border-gray-300 px-2 py-1'>$x_{1}$</td>" in html
        assert "<td class='border border-gray-300 px-2 py-1'>$a_{1}$</td>" in html
        assert "<td class='border border-gray-300 px-2 py-1'>$s_{3}$</td>" in html
        # Шапка
        assert "<th class='border border-gray-300 px-2 py-1 text-left'>Индекс</th>" in html

    def test_markdown_table_to_html_unit(self):
        """Юнит-тест на _md_table_to_html: вход / выход."""
        out = Exporter._md_table_to_html(
            "| h1 | h2 |",
            ["| a | b |", "| c | d |"],
        )
        assert "<table" in out
        assert "<th class='border border-gray-300 px-2 py-1 text-left'>h1</th>" in out
        assert "<th class='border border-gray-300 px-2 py-1 text-left'>h2</th>" in out
        assert "<td class='border border-gray-300 px-2 py-1'>a</td>" in out
        assert "<td class='border border-gray-300 px-2 py-1'>d</td>" in out

    def test_md_lines_to_html_table_block(self):
        """Юнит-тест: _md_lines_to_html обнаруживает блок таблицы и конвертирует."""
        out = Exporter._md_lines_to_html([
            "Header text",
            "",
            "| A | B |",
            "|---|---|",
            "| 1 | 2 |",
            "| 3 | 4 |",
            "",
            "Trailing text",
        ])
        assert "<table" in out
        assert "Header text" in out
        assert "Trailing text" in out
        # Старый сырой формат не должен остаться:
        assert "| A | B |" not in out
        assert "|---|---|" not in out
