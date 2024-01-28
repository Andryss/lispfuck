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
        debug = os.path.join(tmpdirname, "debug")

        with open(source, "w", encoding="utf-8") as file:
            file.write(golden["in_source"])
        with open(input_stream, "w", encoding="utf-8") as file:
            file.write(golden["in_stdin"])

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            with open(source, encoding="utf-8") as s, open(target, "wb") as t, open(debug, "w", encoding="utf-8") as d:
                translator.main(s, t, d)

            with open(debug, encoding="utf-8") as d:
                debug_content = d.read()
            assert debug_content == golden.out["out_debug"]

            with open(target, "rb") as t, open(input_stream, encoding="utf-8") as i:
                machine.main(t, i)

        assert stdout.getvalue() == golden.out["out_stdout"]
