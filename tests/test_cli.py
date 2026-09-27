import pytest

from tide import cli

def test_help_names_the_program(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    assert "usage: tide" in capsys.readouterr().out
