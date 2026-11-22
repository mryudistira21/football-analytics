import json
from pathlib import Path


def test_track_b_colab_notebook_is_valid_json_and_python():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "notebooks" / "track_b_colab.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert notebook["cells"]

    code_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "code"
    ]
    assert code_cells

    for index, cell in enumerate(code_cells):
        source = cell.get("source", [])
        code = "".join(source) if isinstance(source, list) else str(source)
        compile(code, f"{notebook_path.name}:cell-{index}", "exec")


def test_match_colab_notebook_cells_compile():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "match_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) >= 6
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        # IPython shell escapes (!cmd) are not Python; everything else must compile.
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("!"))
        compile(code, f"{path.name}:cell-{index}", "exec")
        assert "import torch" not in source, "the notebook kernel must not load PyTorch"


def test_kaggle_notebook_cells_compile():
    path = Path(__file__).resolve().parents[1] / "notebooks" / "match_kaggle.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [c for c in notebook["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) == 4
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        compile(source, f"{path.name}:cell-{index}", "exec")
        assert "import torch" not in source
