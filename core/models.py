from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple
from fractions import Fraction

# Тип границы переменной: None означает отсутствие ограничения (±∞)
Bound = Optional[Fraction]

# Тип канонического столбца. См. CanonicalVar.
CanonicalVarKind = Literal[
    'orig',         # столбец исходной переменной x_j (≥ 0, возможно после сдвига)
    'slack',        # балансовая переменная s_i (для ограничения <=)
    'surplus',      # избыточная переменная s_i (для ограничения >=)
    'artificial',   # искусственная переменная a_k (для = или >=)
    'split+',       # положительная часть свободной переменной x_j^+
    'split-',       # отрицательная часть свободной переменной x_j^-
]


@dataclass
class CanonicalVar:
    """Метаданные одного столбца расширенной канонической задачи Ax = b, x ≥ 0.

    Используется в [`Exporter`](core/exporter.py) для:
        - вывода блока «Каноническая форма» с осмысленными именами s_i, a_k и т.п.;
        - таблицы соответствия «индекс в расширенной задаче ↔ обозначение ↔ тип»;
        - подстановки display_label в подписи базиса каждого шага.

    Attributes:
        kind: тип столбца (см. :data:`CanonicalVarKind`).
        ext_index: 0-based позиция столбца в расширенной матрице A.
        constraint_row: для slack/surplus/artificial — 0-based номер исходной
            строки, к которой привязана добавленная переменная; иначе None.
        orig_index: для orig/split± — 0-based номер исходной переменной; иначе None.
        display_label: LaTeX-метка для рендеринга (без обрамляющих ``$``).
            Примеры: ``"x_{1}"``, ``"s_{2}"``, ``"a_{1}"``, ``"x_{1}^{+}"``.
    """

    kind: CanonicalVarKind
    ext_index: int
    display_label: str
    constraint_row: Optional[int] = None
    orig_index: Optional[int] = None

@dataclass
class LinearProblem:
    c: List[Fraction]
    A: List[List[Fraction]]
    b: List[Fraction]
    signs: List[str]  # '<=', '>=', '='
    is_max: bool = True
    # Границы переменных: lower_bounds[j] = l_j, upper_bounds[j] = u_j.
    # None означает отсутствие ограничения (-∞ / +∞).
    # По умолчанию все переменные >= 0 (lower=0, upper=None).
    lower_bounds: Optional[List[Bound]] = None
    upper_bounds: Optional[List[Bound]] = None

    def __post_init__(self):
        n = len(self.c)
        m = len(self.A)
        if any(len(row) != n for row in self.A):
            raise ValueError(f"A: все строки должны иметь длину {n}")
        if len(self.b) != m:
            raise ValueError(f"len(b)={len(self.b)} != len(A)={m}")
        if len(self.signs) != m:
            raise ValueError(f"len(signs)={len(self.signs)} != len(A)={m}")
        for s in self.signs:
            if s not in ('<=', '>=', '='):
                raise ValueError(f"signs: ожидается '<=', '>=' или '=', получено: {s!r}")
        if self.lower_bounds is not None and len(self.lower_bounds) != n:
            raise ValueError(f"len(lower_bounds)={len(self.lower_bounds)} != len(c)={n}")
        if self.upper_bounds is not None and len(self.upper_bounds) != n:
            raise ValueError(f"len(upper_bounds)={len(self.upper_bounds)} != len(c)={n}")
        if self.lower_bounds is not None and self.upper_bounds is not None:
            for j, (l, u) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
                if l is not None and u is not None and l > u:
                    raise ValueError(f"lb[{j}]={l} > ub[{j}]={u} (несовместная граница)")

@dataclass
class SimplexStep:
    iteration: int
    N: List[int]            # 0-indexed indices of basic variables (in extended canonical space)
    B_inv: List[List[Fraction]]  # inverse basis matrix (m x m)
    x_B: List[Fraction]     # values of basic variables (length m)
    u_0: List[Fraction]     # dual estimates of current phase
    x_full: List[Fraction] = field(default_factory=list)  # full plan vector x (length n_vars)
    phase: int = 2          # 1 = вспомогательная задача, 2 = основная
    is_optimal: bool = False
    is_unbounded: bool = False
    is_infeasible: bool = False  # фаза I завершилась с w_min > 0
    j_0: Optional[int] = None
    z_0: Optional[List[Fraction]] = None  # length m (basic decomposition)
    t_0: Optional[Fraction] = None
    s_0: Optional[int] = None             # extended index выводимой переменной
    description: str = ""
    c_B: Optional[List[Fraction]] = None
    diffs: Optional[List[tuple]] = None   # список (j, Δ_j) — посчитанные до первого Δ<0 включительно
    ratios: Optional[List[Optional[Fraction]]] = None
    artificial_indices: Optional[List[int]] = None  # индексы искусственных переменных в расширенной задаче
    # Карта восстановления исходных переменных из расширенных (для финального шага)
    var_map: Optional[List[Tuple[str, int, Fraction]]] = None
    # u_0 для исходной (не приведённой) задачи — с учётом инверсий строк
    u_0_original: Optional[List[Fraction]] = None
    # Знаки строк после приведения b>=0 (True = строка была инвертирована)
    row_inverted: Optional[List[bool]] = None
    # Список ошибок валидации восстановленного решения (пустой/None = ОК).
    # Заполняется только на финальном (оптимальном) шаге фазы II.
    validation_errors: Optional[List[str]] = None
    # Исходная базисная матрица (m x m) — столбцы расширенной A с индексами из N
    # в порядке N. Заполняется солвером для возможности её отдельного рендеринга
    # перед обратной матрицей. См. дефект D4 в FIX_PLAN.md.
    B_orig: Optional[List[List[Fraction]]] = None
    # Направление оптимизации ИСХОДНОЙ задачи (True = max). Нужно для управления
    # выводом приоритетного u^* на финальном шаге (D5). Внутри солвера всё всегда
    # сводится к max, и u_0 относится к этой max-форме.
    is_max_problem_original: Optional[bool] = None
