import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from exapt import main


class CLITests(unittest.TestCase):
    def test_audit_prints_json(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = Path(directory) / "kg.csv"
            graph.write_text(
                "x_type,x_id,relation,y_type,y_id\n"
                "drug,1,indication,disease,2\n"
            )

            output = io.StringIO()
            with patch("sys.argv", ["exapt", "audit", str(graph)]):
                with redirect_stdout(output):
                    main()

        self.assertEqual(json.loads(output.getvalue())["rows"], 1)
