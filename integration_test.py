import contextlib
import io
import os
import tempfile

import machine
import pytest
import translator


@pytest.mark.golden_test("golden/*.yml")
def test_translator_and_machine(golden, caplog):
    with tempfile.TemporaryDirectory() as tmpdirname:
        source = os.path.join(tmpdirname, "source")
        input_stream = os.path.join(tmpdirname, "input")
        target = os.path.join(tmpdirname, "target")

        with open(source, "w", encoding="utf-8") as file:
            file.write(golden["in_source"])
        with open(input_stream, "w", encoding="utf-8") as file:
            file.write(golden["in_stdin"])

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            translator.main(source, target)
            machine.main(target, input_stream)

        assert stdout.getvalue() == golden.out["out_stdout"]
