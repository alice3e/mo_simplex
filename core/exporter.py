import re
from typing import List, Optional, Tuple
from fractions import Fraction
from core.models import CanonicalVar, LinearProblem, SimplexStep


class Exporter:
    # ---------------------------------------------------------------- LaTeX helpers
    @staticmethod
    def frac_to_latex(f: Fraction) -> str:
        if f.denominator == 1:
            return str(f.numerator)
        # \dfrac (display-style) занимает больше вертикального пространства,
        # что предотвращает визуальное наложение знаменателя на следующую строку
        # в матрицах bmatrix при использовании KaTeX.
        if f.numerator < 0:
            return f"-\\dfrac{{{-f.numerator}}}{{{f.denominator}}}"
        return f"\\dfrac{{{f.numerator}}}{{{f.denominator}}}"

    @staticmethod
    def _has_frac(items: List[Fraction]) -> bool:
        """Проверяет, есть ли в списке хотя бы одна нецелая дробь."""
        return any(x.denominator != 1 for x in items)

    @staticmethod
    def _dual_var_sign(prim_sign: str, is_max: bool) -> str:
        """Возвращает LaTeX-знак для двойственной переменной u_i.

        Зависит от типа i-го прямого ограничения и направления оптимизации.

        Для прямой `min`:
            ``=`` → u_i свободная;
            ``>=`` → u_i ≥ 0;
            ``<=`` → u_i ≤ 0.

        Для прямой `max` правила инвертированы:
            ``=`` → u_i свободная;
            ``<=`` → u_i ≥ 0;
            ``>=`` → u_i ≤ 0.
        """
        if prim_sign == '=':
            return '\\in \\mathbb{R}'
        if is_max:
            return '\\geq 0' if prim_sign == '<=' else '\\leq 0'
        return '\\geq 0' if prim_sign == '>=' else '\\leq 0'

    @staticmethod
    def _dual_constr_sign(j: int, problem: 'LinearProblem') -> str:
        """Возвращает LaTeX-знак для j-го двойственного ограничения.

        Зависит от типа j-й прямой переменной x_j и направления:
            свободная (lb=None, ub=None)            → ``=``;
            x_j ≤ 0 (lb=None, ub=Fraction(0))       → ``\\leq`` (для min), ``\\geq`` (для max);
            стандартная / x_j ≥ lb / x_j ≤ ub и т.п. → ``\\geq`` (для min), ``\\leq`` (для max).

        Замечание: проект пока поддерживает фактически только x_j ≥ lb (с возможной
        верхней границей). Случай x_j ≤ 0 (lb=None, ub=0) включён для будущей
        расширяемости.
        """
        lb = problem.lower_bounds[j] if problem.lower_bounds else Fraction(0)
        ub = problem.upper_bounds[j] if problem.upper_bounds else None

        is_free = (lb is None) and (ub is None)
        is_neg_only = (lb is None) and (ub is not None) and (ub == Fraction(0))

        if is_free:
            return '='
        if problem.is_max:
            return '\\leq' if is_neg_only else '\\geq'
        return '\\geq' if is_neg_only else '\\leq'

    @staticmethod
    def vec_to_latex(v: List[Fraction], is_column: bool = True) -> str:
        if is_column:
            # Если есть дроби — добавляем [2ex] после \\ для увеличения межстрочного интервала,
            # чтобы знаменатель \dfrac не накладывался на следующую строку в KaTeX.
            sep = " \\\\[2ex] " if Exporter._has_frac(v) else " \\\\ "
            lines = sep.join(Exporter.frac_to_latex(x) for x in v)
        else:
            lines = " & ".join(Exporter.frac_to_latex(x) for x in v)
        return f"\\begin{{bmatrix}} {lines} \\end{{bmatrix}}"

    @staticmethod
    def mat_to_latex(M: List[List[Fraction]]) -> str:
        # Если в матрице есть дроби — добавляем [2ex] после \\ для увеличения межстрочного интервала.
        all_flat = [x for row in M for x in row]
        sep = " \\\\[2ex]\n" if Exporter._has_frac(all_flat) else " \\\\ \n"
        rows = [" & ".join(Exporter.frac_to_latex(x) for x in row) for row in M]
        body = sep.join(rows)
        return f"\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}"

    # ---------------------------------------------------------------- problem statements
    @staticmethod
    def _format_objective_terms(coeffs: List[Fraction], var_letter: str = "x") -> str:
        terms = []
        for i, c in enumerate(coeffs):
            if c != 0:
                is_first = not terms
                if c < 0:
                    sign = "-"
                elif is_first:
                    sign = ""
                else:
                    sign = "+"
                c_abs = abs(c)
                var = f"{var_letter}_{{{i+1}}}"
                if c_abs == 1:
                    coeff_str = ""
                    sep = ""
                else:
                    coeff_str = Exporter.frac_to_latex(c_abs)
                    # Если коэффициент — дробь (\dfrac{...}{...}), добавляем тонкий пробел
                    # чтобы знаменатель не склеивался с именем переменной
                    sep = "\\," if coeff_str.startswith("\\dfrac") else ""
                terms.append(f"{sign} {coeff_str}{sep}{var}")
        if not terms:
            terms.append("0")
        s = " ".join(terms).strip()
        if s.startswith("+ "):
            s = s[2:]
        return s

    @staticmethod
    def _format_problem_block(problem: LinearProblem) -> List[str]:
        lines: List[str] = []
        lines.append("$$")
        lines.append("\\begin{align*}")
        obj_str = Exporter._format_objective_terms(problem.c, "x")
        target = "\\max" if problem.is_max else "\\min"
        lines.append(f"{obj_str} &\\to {target} \\\\")
        for i, row in enumerate(problem.A):
            row_str = Exporter._format_objective_terms(row, "x")
            sign_map = {"<=": "\\leq", ">=": "\\geq", "=": "="}
            lines.append(f"{row_str} &{sign_map[problem.signs[i]]} {Exporter.frac_to_latex(problem.b[i])} \\\\")
        lines.append("x &\\geq 0")
        lines.append("\\end{align*}")
        lines.append("$$\n")
        return lines

    @staticmethod
    def _format_dual_block(problem: LinearProblem) -> List[str]:
        """Рендерит двойственную задачу с КОРРЕКТНЫМИ знаками.

        Знаки двойственных ограничений (по столбцам ``j``) определяются типом
        исходной переменной ``x_j`` ([`_dual_constr_sign`](core/exporter.py)).
        Знаки двойственных переменных ``u_i`` (по строкам ``i``) определяются
        типом исходного ограничения ([`_dual_var_sign`](core/exporter.py)).

        Это устраняет регрессию, при которой во всех случаях выводилась единственная
        строка ``u ≥ 0`` независимо от исходной задачи.
        """
        lines: List[str] = []
        lines.append("$$")
        lines.append("\\begin{align*}")
        dual_obj_str = Exporter._format_objective_terms(problem.b, "u")
        dual_target = "\\min" if problem.is_max else "\\max"
        lines.append(f"{dual_obj_str} &\\to {dual_target} \\\\")
        for j in range(len(problem.c)):
            col = [problem.A[i][j] for i in range(len(problem.A))]
            row_str = Exporter._format_objective_terms(col, "u")
            sign_str = Exporter._dual_constr_sign(j, problem)
            lines.append(f"{row_str} &{sign_str} {Exporter.frac_to_latex(problem.c[j])} \\\\")
        m = len(problem.A)
        for i in range(m):
            sign = Exporter._dual_var_sign(problem.signs[i], problem.is_max)
            lines.append(f"u_{{{i+1}}} &{sign} \\\\")
        lines.append("\\end{align*}")
        lines.append("$$\n")
        return lines

    # ---------------------------------------------------------------- canonical form
    @staticmethod
    def _format_canonical_block(
        problem: LinearProblem,
        canonical_vars: List[CanonicalVar],
        canonical_A: List[List[Fraction]],
        canonical_b: List[Fraction],
        canonical_signs_after_norm: List[str],
        artificial_indices: List[int],
    ) -> List[str]:
        """Рендерит блок «Каноническая форма» в Markdown/LaTeX.

        Включает:
            1. Систему ``A_canonical · x = b``, ``x ≥ 0`` (после редукций
               и нормализации ``b ≥ 0``), c осмысленными метками
               из :class:`core.models.CanonicalVar`.
            2. Таблицу соответствия «индекс в расширенной задаче ↔ обозначение ↔ тип».
            3. Целевую функцию фазы I (если есть искусственные).

        Args:
            problem: исходная задача (используется для `_format_objective_terms`).
            canonical_vars: метаданные расширенных столбцов (см. ``solver.canonical_vars``).
            canonical_A: расширенная матрица системы ``A`` (после всех редукций).
            canonical_b: вектор правой части ``b`` (после нормализации).
            canonical_signs_after_norm: знаки строк после ``b ≥ 0`` (для справки).
            artificial_indices: ext_index искусственных переменных
                (для целевой функции фазы I).
        """
        lines: List[str] = []
        m = len(canonical_b)

        # --- 1. Система уравнений ---
        lines.append("$$")
        lines.append("\\begin{align*}")
        for i in range(m):
            row_terms: List[str] = []
            for cv in canonical_vars:
                coeff = canonical_A[i][cv.ext_index]
                if coeff == 0:
                    continue
                is_first = not row_terms
                if coeff < 0:
                    sign = '-'
                elif is_first:
                    sign = ''
                else:
                    sign = '+'
                absv = abs(coeff)
                if absv == 1:
                    coeff_str = ''
                    sep = ''
                else:
                    coeff_str = Exporter.frac_to_latex(absv)
                    sep = '\\,' if coeff_str.startswith('\\dfrac') else ''
                row_terms.append(f"{sign} {coeff_str}{sep}{cv.display_label}")
            if not row_terms:
                row_terms.append("0")
            eq = ' '.join(row_terms).strip()
            if eq.startswith('+ '):
                eq = eq[2:]
            lines.append(f"{eq} &= {Exporter.frac_to_latex(canonical_b[i])} \\\\")
        # Список нестандартных «всех переменных ≥ 0» — без указания конкретных букв,
        # чтобы не загромождать формулу.
        lines.append("\\text{все переменные} &\\geq 0")
        lines.append("\\end{align*}")
        lines.append("$$\n")

        # --- 2. Таблица соответствия ---
        non_orig = [cv for cv in canonical_vars if cv.kind != 'orig']
        if non_orig:
            lines.append("\n**Соответствие расширенных переменных:**\n")
            lines.append("| Индекс | Обозначение | Тип |")
            lines.append("|---|---|---|")
            kind_ru = {
                'orig': 'исходная',
                'slack': 'балансовая (для $\\leq$)',
                'surplus': 'избыточная (для $\\geq$)',
                'artificial': 'искусственная',
                'split+': 'положит. часть свободной',
                'split-': 'отриц. часть свободной',
            }
            for cv in canonical_vars:
                lines.append(
                    f"| $x_{{{cv.ext_index + 1}}}$ | "
                    f"${cv.display_label}$ | "
                    f"{kind_ru.get(cv.kind, cv.kind)} |"
                )
            lines.append("")

        # --- 3. Целевая функция фазы I (если есть искусственные) ---
        if artificial_indices:
            ext_to_label = {cv.ext_index: cv.display_label for cv in canonical_vars}
            art_labels = [
                ext_to_label.get(k, f"x_{{{k + 1}}}") for k in artificial_indices
            ]
            w_str = ' + '.join(art_labels)
            lines.append(
                f"\n**Целевая функция фазы I:** "
                f"$w = {w_str} \\to \\min$\n"
            )

        return lines

    # ---------------------------------------------------------------- step rendering
    @staticmethod
    def _phase_label(phase: int) -> str:
        return "Фаза I (вспомогательная задача)" if phase == 1 else "Фаза II"

    @staticmethod
    def _render_step_lines(
        step: SimplexStep,
        detailed: bool,
        final_answer: Optional[Fraction],
        x_original: Optional[List[Fraction]] = None,
        n_orig_vars: Optional[int] = None,
        canonical_vars: Optional[List[CanonicalVar]] = None,
    ) -> List[str]:
        """Возвращает линии Markdown-описания шага (без HTML-обёрток)."""
        lines: List[str] = []

        artificial_set = set(step.artificial_indices or [])

        # Карта ext_index → осмысленная метка (s_i / a_k / x_j^+ ...).
        # Используется, если caller передал ``canonical_vars`` (см. Этап 7).
        label_map = (
            {cv.ext_index: cv.display_label for cv in canonical_vars}
            if canonical_vars else None
        )

        def var_label(idx: int) -> str:
            if label_map is not None and idx in label_map:
                return label_map[idx]
            if idx in artificial_set:
                return f"y_{{{idx+1}}}"
            return f"x_{{{idx+1}}}"

        basis_labels = ", ".join(var_label(n) for n in step.N)
        lines.append(f"**Базис $N$:** $({basis_labels})$  ")
        lines.append(f"**Полный план $x$:**  ")
        lines.append(f"$$ x = {Exporter.vec_to_latex(step.x_full, is_column=False)} $$  ")
        lines.append(f"**Базисная подвыборка $x_B$:**  ")
        lines.append(f"$$ x_B = {Exporter.vec_to_latex(step.x_B)} $$  ")

        # Исходная базисная матрица B_orig (по столбцам A для индексов из N).
        # Выводится ПЕРЕД обратной — см. замечание профессора и D4 в FIX_PLAN.md.
        if step.B_orig is not None:
            lines.append(
                "**Исходная базисная матрица $B_{\\text{orig}}$ "
                "(столбцы $A_j$ для $j \\in N$):**  "
            )
            lines.append(
                f"$$ B_{{\\text{{orig}}}} = "
                f"{Exporter.mat_to_latex(step.B_orig)} $$  "
            )

        lines.append(f"**Обратная базисная матрица $B$:**  ")
        lines.append(f"$$ B = {Exporter.mat_to_latex(step.B_inv)} $$  ")

        # Пометка о порядке строк B (D6 в FIX_PLAN.md).
        if step.N:
            mapping = ", ".join(
                f"строка {i+1} → ${var_label(idx)}$"
                for i, idx in enumerate(step.N)
            )
            lines.append(
                "*Строка $i$ матрицы $B$ соответствует $i$-му исходному "
                "ограничению, не $i$-й базисной переменной. Соответствие "
                f"строк и базисных переменных: {mapping}.*  "
            )

        if detailed and step.c_B is not None:
            c_b_str = "\\begin{bmatrix} " + " & ".join(Exporter.frac_to_latex(x) for x in step.c_B) + " \\end{bmatrix}"
            lines.append(
                f"**Вектор оценок $u_0 = c_B B$:**  \n"
                f"$$ u_0 = {c_b_str} \\cdot B = {Exporter.vec_to_latex(step.u_0, is_column=False)} $$  "
            )
        else:
            lines.append(f"**Вектор оценок $u_0$:**  \n$$ u_0 = {Exporter.vec_to_latex(step.u_0, is_column=False)} $$  ")

        # Δ_j: выводим только посчитанные (до первого отрицательного включительно).
        if step.diffs:
            lines.append("**Проверка оптимальности $\\Delta_j = u_0 A_j - c_j$ (по правилу первого индекса):**")
            lines.append("$$ \\begin{align*} ")
            for j, diff in step.diffs:
                marker = ""
                if diff < 0:
                    marker = " \\quad <\\!0\\ \\Rightarrow\\ j_0 = " + str(j + 1)
                lines.append(f"\\Delta_{{{j+1}}} &= {Exporter.frac_to_latex(diff)}{marker} \\\\ ")
            lines.append("\\end{align*} $$\n")

        # Финальные исходы шага
        if step.is_infeasible:
            lines.append("\n❌ **Задача несовместна.** На фазе I минимум суммы искусственных переменных строго положителен.\n")
            return lines

        if step.is_optimal:
            if step.phase == 1:
                lines.append("\n✓ **Фаза I завершена.** Все искусственные равны нулю — переходим к фазе II.\n")
            else:
                lines.append("\n✓ **План оптимален.** Все двойственные ограничения выполнены.\n")
                if final_answer is not None:
                    lines.append(f"**Ответ:** $f^* = {Exporter.frac_to_latex(final_answer)}$\n")

                # Восстановленные исходные переменные (если были редукции)
                if x_original is not None and n_orig_vars is not None:
                    x_orig_str = Exporter.vec_to_latex(x_original[:n_orig_vars], is_column=False)
                    lines.append(f"**Исходные переменные $x^*$:** $x^* = {x_orig_str}$\n")

                # Раздел двойственного решения с учётом знаков
                lines.extend(Exporter._render_dual_solution(step))

            return lines

        if step.is_unbounded:
            lines.append("\n❌ **Целевая функция не ограничена.** Решения нет.\n")
            return lines

        # Обычный шаг: ввод/вывод
        j0_label = step.j_0 + 1 if step.j_0 is not None else "?"
        lines.append(f"\n**План не оптимален.** Вводим в базис $x_{{{j0_label}}}$.\n")
        if step.z_0 is not None:
            if detailed:
                lines.append(
                    f"**Направляющий вектор $z_0 = B \\cdot A_{{{j0_label}}}$:**  \n"
                    f"$$ z_0 = {Exporter.vec_to_latex(step.z_0)} $$  "
                )
            else:
                lines.append(f"**Направляющий вектор $z_0$:**  \n$$ z_0 = {Exporter.vec_to_latex(step.z_0)} $$  ")

        if step.ratios is not None and step.t_0 is not None and step.z_0 is not None:
            if detailed:
                ratios_str = ", ".join(
                    f"\\dfrac{{{Exporter.frac_to_latex(step.x_B[i])}}}{{{Exporter.frac_to_latex(step.z_0[i])}}}"
                    for i, r in enumerate(step.ratios) if r is not None
                )
                lines.append(f"**Шаг $t_0$:** $t_0 = \\min\\left\\{{{ratios_str}\\right\\}} = {Exporter.frac_to_latex(step.t_0)}$  ")
            else:
                lines.append(f"**Шаг $t_0$:** $t_0 = {Exporter.frac_to_latex(step.t_0)}$  ")
        if step.s_0 is not None:
            lines.append(f"Выводим из базиса $x_{{{step.s_0+1}}}$.\n")
        return lines

    @staticmethod
    def _render_dual_solution(step: SimplexStep) -> List[str]:
        """Раздел двойственного решения для исходной задачи.

        Основной показываемый вектор — :attr:`SimplexStep.u_0_original`
        (с правильными знаками для исходной задачи: учтены направление
        оптимизации и инверсии строк при ``b < 0``). Внутренний ``u_0``
        для приведённой ``max``-формы выводится только как справочный блок,
        и только если он действительно отличается от ``u_0_original``
        (т.е. для ``min``-задачи или при наличии инверсий).
        """
        lines: List[str] = []
        lines.append("---\n")
        lines.append("**Двойственное решение $u^*$:**\n")

        u_orig = step.u_0_original
        if u_orig is not None:
            # Основной вектор — для исходной задачи.
            lines.append(
                f"$$ u^* = {Exporter.vec_to_latex(u_orig, is_column=False)} $$\n"
            )

            row_inv = step.row_inverted or []
            has_inversions = any(row_inv)
            # Признак, что внутренний u_0 отличается знаками от u_0_original
            # (либо из-за инверсий строк, либо из-за min-задачи).
            differs_from_internal = (
                has_inversions
                or step.is_max_problem_original is False
            )

            if has_inversions:
                inv_indices = [i + 1 for i, inv in enumerate(row_inv) if inv]
                inv_str = ", ".join(str(k) for k in inv_indices)
                lines.append(
                    f"*Знаки $u_i$ для строк {inv_str} инвертированы обратно: "
                    f"эти строки были умножены на $-1$ при приведении "
                    f"$b \\geq 0$.*\n"
                )
            if step.is_max_problem_original is False:
                lines.append(
                    "*Для исходной задачи на $\\min$ знак двойственных переменных "
                    "противоположен оценкам, полученным во внутренней "
                    "$\\max$-форме.*\n"
                )

            if differs_from_internal:
                # Служебный блок: внутренние оценки приведённой max-формы.
                lines.append(
                    "<details><summary>Внутренние оценки приведённой "
                    "$\\max$-формы $u_0$ (справочно)</summary>\n"
                )
                lines.append(
                    f"$$ u_0 = "
                    f"{Exporter.vec_to_latex(step.u_0, is_column=False)} $$\n"
                )
                lines.append("</details>\n")
        else:
            # Fallback (старое поведение): нет u_0_original — показываем u_0.
            lines.append(
                f"$$ u^* = u_0 = "
                f"{Exporter.vec_to_latex(step.u_0, is_column=False)} $$\n"
            )

        lines.append(
            "**Проверка сильной двойственности:** "
            "$f^* = c^T x^* = b^T u^*$\n"
        )
        return lines

    # ---------------------------------------------------------------- markdown
    @staticmethod
    def generate_markdown(
        problem: LinearProblem,
        steps: List[SimplexStep],
        final_answer: Optional[Fraction],
        detailed: bool = False,
        hidden_steps: Optional[List[int]] = None,
        x_original: Optional[List[Fraction]] = None,
        n_orig_vars: Optional[int] = None,
        # --- НОВОЕ (Этап 4): блок «Каноническая форма» ---
        canonical_vars: Optional[List[CanonicalVar]] = None,
        canonical_A: Optional[List[List[Fraction]]] = None,
        canonical_b: Optional[List[Fraction]] = None,
        canonical_signs_after_norm: Optional[List[str]] = None,
        artificial_indices: Optional[List[int]] = None,
    ) -> str:
        hidden = set(hidden_steps or [])
        md: List[str] = []
        md.append("# Решение задачи модифицированным двухфазным симплекс-методом\n")

        md.append("## Прямая задача")
        md.extend(Exporter._format_problem_block(problem))

        # Опциональный блок «Каноническая форма» — между прямой и двойственной.
        if (
            canonical_vars is not None
            and canonical_A is not None
            and canonical_b is not None
        ):
            md.append("## Каноническая форма")
            md.extend(Exporter._format_canonical_block(
                problem,
                canonical_vars,
                canonical_A,
                canonical_b,
                canonical_signs_after_norm or [],
                artificial_indices or [],
            ))

        md.append("## Двойственная задача")
        md.extend(Exporter._format_dual_block(problem))

        md.append("---\n")

        last_phase: Optional[int] = None
        for step in steps:
            if step.iteration in hidden:
                continue
            if step.phase != last_phase:
                md.append(f"\n## {Exporter._phase_label(step.phase)}\n")
                last_phase = step.phase

            is_final = step.is_optimal and step.phase == 2
            step_title = "### Итог" if is_final else f"### Шаг {step.iteration}"
            md.append(step_title)
            md.extend(Exporter._render_step_lines(
                step, detailed, final_answer,
                x_original=x_original,
                n_orig_vars=n_orig_vars,
                canonical_vars=canonical_vars,
            ))
            md.append("---\n")

        return "\n".join(md)

    # ---------------------------------------------------------------- html
    @staticmethod
    def generate_html(
        problem: LinearProblem,
        steps: List[SimplexStep],
        final_answer: Optional[Fraction],
        detailed: bool = False,
        hidden_steps: Optional[List[int]] = None,
        x_original: Optional[List[Fraction]] = None,
        n_orig_vars: Optional[int] = None,
        # --- НОВОЕ (Этап 4): блок «Каноническая форма» ---
        canonical_vars: Optional[List[CanonicalVar]] = None,
        canonical_A: Optional[List[List[Fraction]]] = None,
        canonical_b: Optional[List[Fraction]] = None,
        canonical_signs_after_norm: Optional[List[str]] = None,
        artificial_indices: Optional[List[int]] = None,
    ) -> str:
        hidden = set(hidden_steps or [])
        html: List[str] = []

        # Шапка: прямая + каноническая + двойственная
        html.append("<div class='mb-8 pb-4 border-b border-gray-200'>")
        html.append("<h3 class='text-xl font-bold text-gray-800 mb-4'>Прямая задача</h3>")
        html.append("<div class='overflow-x-auto bg-gray-50 p-4 rounded-lg'>")
        html.extend(Exporter._format_problem_block(problem))
        html.append("</div>")

        if (
            canonical_vars is not None
            and canonical_A is not None
            and canonical_b is not None
        ):
            html.append(
                "<h3 class='text-xl font-bold text-gray-800 mb-4 mt-6'>Каноническая форма</h3>"
            )
            html.append("<div class='overflow-x-auto bg-gray-50 p-4 rounded-lg'>")
            canonical_lines = Exporter._format_canonical_block(
                problem,
                canonical_vars,
                canonical_A,
                canonical_b,
                canonical_signs_after_norm or [],
                artificial_indices or [],
            )
            # Каноническая форма содержит и LaTeX-блоки `$$ ... $$`, и markdown-таблицу,
            # и текстовые подписи — пропускаем через тот же markdown→HTML конвертер,
            # что и шаги, иначе **жирный**/таблицы не отрендерятся.
            html.append(Exporter._md_lines_to_html(canonical_lines))
            html.append("</div>")

        html.append("<h3 class='text-xl font-bold text-gray-800 mb-4 mt-6'>Двойственная задача</h3>")
        html.append("<div class='overflow-x-auto bg-gray-50 p-4 rounded-lg'>")
        html.extend(Exporter._format_dual_block(problem))
        html.append("</div>")
        html.append("</div>")

        last_phase: Optional[int] = None
        for step in steps:
            if step.iteration in hidden:
                continue

            if step.phase != last_phase:
                phase_color = "indigo" if step.phase == 2 else "amber"
                html.append(
                    f"<h3 class='text-2xl font-bold text-{phase_color}-700 mt-8 mb-4 border-b-2 border-{phase_color}-300 pb-1'>"
                    f"{Exporter._phase_label(step.phase)}</h3>"
                )
                last_phase = step.phase

            html.append(
                f"<div class='step-container bg-white p-6 rounded-lg border border-gray-200 shadow-sm mb-6' "
                f"id='step-card-{step.iteration}'>"
            )
            html.append("<div class='flex justify-between items-center border-b pb-2 mb-4'>")
            html.append(f"<h4 class='text-lg font-bold text-indigo-700'>Шаг {step.iteration} ({Exporter._phase_label(step.phase)})</h4>")
            html.append(
                f"<label class='text-sm text-gray-500 flex items-center cursor-pointer no-print'>"
                f"<input type='checkbox' class='mr-2 step-visibility-toggle' data-step='{step.iteration}' "
                f"checked onchange='toggleStepVisibility(this)'> Показывать шаг</label>"
            )
            html.append("</div>")

            html.append("<div class='space-y-2 overflow-x-auto'>")
            md_lines = Exporter._render_step_lines(
                step, detailed, final_answer,
                x_original=x_original,
                n_orig_vars=n_orig_vars,
                canonical_vars=canonical_vars,
            )
            html.append(Exporter._md_lines_to_html(md_lines))
            html.append("</div>")
            html.append("</div>")

        return "\n".join(html)

    # ---------------------------------------------------------------- markdown → HTML
    @staticmethod
    def _md_lines_to_html(md_lines: List[str]) -> str:
        """Преобразует список markdown-строк (с inline `$...$` и блочными `$$...$$`)
        в HTML, сохраняя многострочные $$-блоки в едином DOM-узле для KaTeX.

        Минимальный markdown-парсинг:
        * `**bold**` → `<strong>bold</strong>`;
        * trailing `  ` перед `\n` → `<br>`;
        * `---` на отдельной строке → `<hr>`;
        * GitHub-flavored таблицы вида ``| h1 | h2 | / |---|---| / | c1 | c2 |``
          → ``<table>``. Inline ``$...$`` внутри ячеек сохраняется, чтобы KaTeX
          мог отрендерить их по auto-render.
        """
        # --- 1. Сначала отделяем и конвертируем markdown-таблицы построчно. ---
        # Любой блок из 2+ подряд идущих строк, начинающихся с '|' и содержащих
        # `|---|...` разделитель во второй строке, превращается в HTML-таблицу.
        out_lines: List[str] = []
        i = 0
        while i < len(md_lines):
            line = md_lines[i]
            stripped = line.strip()
            # Кандидат заголовка: непустая, начинается с '|'.
            if (
                stripped.startswith('|')
                and i + 1 < len(md_lines)
                and re.fullmatch(r"\s*\|(\s*:?-+:?\s*\|)+\s*", md_lines[i + 1])
            ):
                # Собираем строки таблицы.
                header = stripped
                # i+1 — разделитель, пропускаем.
                j = i + 2
                rows: List[str] = []
                while j < len(md_lines) and md_lines[j].strip().startswith('|'):
                    rows.append(md_lines[j].strip())
                    j += 1
                out_lines.append(Exporter._md_table_to_html(header, rows))
                i = j
                continue
            out_lines.append(line)
            i += 1

        text = "\n".join(out_lines)

        # --- 2. inline-форматирование ---
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.DOTALL)
        text = re.sub(r"(?m)^---\s*$", "<hr class='my-3 border-gray-200'>", text)
        text = re.sub(r"  +\n", "<br>\n", text)
        return text

    @staticmethod
    def _md_table_to_html(header_line: str, body_lines: List[str]) -> str:
        """Конвертирует GitHub-flavored md-таблицу в HTML-таблицу с Tailwind-стилями.

        Inline ``$...$``-формулы внутри ячеек оставляем без изменений — их
        отрендерит KaTeX auto-render на стороне браузера.
        """
        def _cells(raw: str) -> List[str]:
            # Убираем крайние '|' и разбиваем по '|' (без обработки экранирования —
            # внутри наших таблиц пайпа в данных нет).
            stripped = raw.strip().strip('|')
            return [c.strip() for c in stripped.split('|')]

        headers = _cells(header_line)
        rows = [_cells(r) for r in body_lines]
        out: List[str] = []
        out.append(
            "<table class='min-w-full text-sm border border-gray-300 my-2'>"
        )
        out.append("<thead class='bg-gray-100'><tr>")
        for h in headers:
            out.append(
                f"<th class='border border-gray-300 px-2 py-1 text-left'>{h}</th>"
            )
        out.append("</tr></thead>")
        out.append("<tbody>")
        for row in rows:
            out.append("<tr>")
            for c in row:
                out.append(
                    f"<td class='border border-gray-300 px-2 py-1'>{c}</td>"
                )
            out.append("</tr>")
        out.append("</tbody></table>")
        return "\n".join(out)
