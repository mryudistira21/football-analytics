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
