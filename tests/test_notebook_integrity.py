"""Static integrity checks for the Kaggle notebooks.

Written after 06_final_training.ipynb shipped with its clone cell replaced by the
markdown heading "## 1 - Clone the repo". As a code cell that is a comment: it ran,
succeeded, and never cloned the repo. Nothing failed until section 8 needed REPO_DIR,
six sections and several minutes of paid GPU later.

Same shape as every failure in docs/04_experiment_register.md -- a step that cannot do
its job still produces a well-formed artefact. The generator had validated that each
cell PARSED, which a comment does. Parsing is not doing anything.

These checks run on the notebook JSON, no kernel required.
"""
import ast, builtins, json, tempfile, unittest
from pathlib import Path

BUILTINS = set(dir(builtins)) | {"get_ipython", "display", "__file__"}

def strip_magics(src):
    out = []
    for line in src.splitlines():
        s = line.lstrip()
        out.append("" if (s.startswith("!") or s.startswith("%")) else line)
    return "\n".join(out)

class Defs(ast.NodeVisitor):
    """Every name this cell binds, at any nesting depth. Permissive on purpose:
    a false 'defined' hides a bug in one cell, a false 'undefined' cries wolf on
    every notebook and the check gets switched off."""
    def __init__(self): self.names = set()
    def visit_Name(self, n):
        if isinstance(n.ctx, (ast.Store, ast.Del)): self.names.add(n.id)
        self.generic_visit(n)
    def visit_alias(self, n): self.names.add((n.asname or n.name).split(".")[0])
    def visit_FunctionDef(self, n): self.names.add(n.name); self.generic_visit(n)
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, n): self.names.add(n.name); self.generic_visit(n)
    def visit_ExceptHandler(self, n):
        if n.name: self.names.add(n.name)
        self.generic_visit(n)
    def visit_arg(self, n): self.names.add(n.arg); self.generic_visit(n)

def loads(tree):
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}

def check(path):
    nb = json.loads(path.read_text())
    problems, defined = [], set()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        raw = "".join(c["source"])
        code = strip_magics(raw)
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            problems.append(f"cell {i}: syntax error: {e}")
            continue
        if not tree.body and raw.strip():
            problems.append(
                f"cell {i}: CODE CELL THAT DOES NOTHING - only comments/whitespace. "
                f"Almost always markdown pasted into a code cell; it runs, succeeds, "
                f"and silently skips whatever the heading promised. "
                f"First line: {raw.strip().splitlines()[0][:60]!r}")
        # A cell's own definitions count: check against earlier cells PLUS this one,
        # or every self-contained cell reports its own imports as undefined.
        d = Defs(); d.visit(tree)
        for name in sorted(loads(tree) - defined - d.names - BUILTINS):
            problems.append(f"cell {i}: uses {name!r}, which no earlier cell defines")
        defined |= d.names
    return problems



class NotebookIntegrity(unittest.TestCase):
    """Every notebook, every cell."""

    def test_no_notebook_has_integrity_problems(self):
        notebooks = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
        self.assertTrue(notebooks, "no notebooks found - wrong path?")
        for nb in notebooks:
            with self.subTest(notebook=nb.name):
                self.assertEqual([], check(nb), f"{nb.name} has integrity problems")

    def test_the_check_catches_a_markdown_heading_in_a_code_cell(self):
        """The bug this file exists for. A check that only ever passes proves nothing."""
        nb = json.loads((Path(__file__).resolve().parents[1]
                         / "notebooks/06_final_training.ipynb").read_text())
        for c in nb["cells"]:
            if c["cell_type"] == "code" and "REPO_DIR = Path(" in "".join(c["source"]):
                c["source"] = ["## 1 - Clone the repo"]
                break
        else:
            self.fail("06_final_training.ipynb no longer defines REPO_DIR in a code cell")

        with tempfile.TemporaryDirectory() as d:
            broken = Path(d) / "broken.ipynb"
            broken.write_text(json.dumps(nb))
            problems = check(broken)

        self.assertTrue(any("DOES NOTHING" in p for p in problems),
                        f"no-op cell not flagged: {problems}")
        self.assertTrue(any("REPO_DIR" in p for p in problems),
                        f"undefined REPO_DIR not flagged: {problems}")

    def test_a_name_defined_only_inside_its_own_cell_is_not_flagged(self):
        """The first version of this check reported every cell's own imports as
        undefined, because it compared against earlier cells only."""
        with tempfile.TemporaryDirectory() as d:
            nb = Path(d) / "selfcontained.ipynb"
            nb.write_text(json.dumps({"cells": [{
                "cell_type": "code", "metadata": {}, "execution_count": None,
                "outputs": [], "source": ["import os\n", "print(os.sep)\n"]}],
                "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
            self.assertEqual([], check(nb))


if __name__ == "__main__":
    unittest.main(verbosity=2)
