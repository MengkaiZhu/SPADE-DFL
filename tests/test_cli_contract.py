import pytest

from spade_dfl_release.cli import FAMILIES, build_parser, main
from spade_dfl_release.runners import execute_family


def test_only_paper_experiment_families_are_exposed():
    assert FAMILIES == ("main", "sensitivity", "heterogeneity", "communication")


def test_cli_rejects_numerical_parameter_overrides():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["main", "--verify-existing", "--tau", "99"])


def test_cli_requires_exactly_one_mode():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["main"])
    with pytest.raises(SystemExit):
        parser.parse_args(["main", "--verify-existing", "--rerun"])


def test_fixed_family_help_names_only_the_selected_family(capsys):
    build_parser("main").print_help()
    help_text = capsys.readouterr().out
    assert "main" in help_text
    assert "sensitivity" not in help_text


def test_cli_rejects_a_rerun_output_directory_that_already_exists(tmp_path):
    with pytest.raises(SystemExit):
        main(["--rerun", "--output", str(tmp_path)], fixed_family="main")


def test_cli_rejects_a_rerun_output_inside_reference_results(reference_root):
    with pytest.raises(SystemExit):
        main(["--rerun", "--output", str(reference_root / "new")], fixed_family="main")


def test_runner_rejects_a_rerun_without_an_output_path():
    with pytest.raises(ValueError, match="requires --output"):
        execute_family("main", "rerun")


def test_runner_rejects_a_rerun_output_directory_that_already_exists(tmp_path):
    with pytest.raises(ValueError, match="new directory"):
        execute_family("main", "rerun", tmp_path)


def test_runner_rejects_a_rerun_output_inside_reference_results(reference_root):
    with pytest.raises(ValueError, match="outside reference_results"):
        execute_family("main", "rerun", reference_root / "new")
