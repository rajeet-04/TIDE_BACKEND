import pytest

from tide import cli

def test_help_names_the_program(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli.main(["--help"])
    assert stopped.value.code == 0
    assert "usage: tide" in capsys.readouterr().out

def test_predict_command_parses():
    args = cli.build_parser().parse_args(["predict", "--port", "haldia", "--start", "2026-10-01", "--hours", "24"])
    assert args.handler.__name__ == "_predict" and args.hours == 24.0 and args.step == 60
