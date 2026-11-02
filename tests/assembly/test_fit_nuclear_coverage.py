"""Reproduced fitting-domain and identifiability failures, without an external R dependency."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/assembly_benchmarks/fit_nuclear_coverage.py'
SPEC = importlib.util.spec_from_file_location('fit_nuclear_coverage', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_low_coverage_signal_is_retained_when_tail_has_lower_counts():
    # The observed Narthecium spectrum: global minimum within15 would trim to15.
    counts = [98152927, 24189772, 22077366, 22645324, 23229530,
              22536580, 20413045, 17322860, 13930232, 10610122,
              7569038, 5351597, 3695638, 2630009, 1898055]
    spectrum = '\n'.join(f'{index} {count}' for index, count in enumerate(counts, 1))
    assert MODULE.first_valley(spectrum) == 2


def test_no_separation_between_error_and_genome_is_not_a_fitted_depth():
    with pytest.raises(ValueError, match='No strict low-frequency valley'):
        MODULE.first_valley('\n'.join(f'{index} {10000//index**2}' for index in range(1, 16)))


def test_missing_frequency_bins_do_not_change_the_meaning_of_the_domain():
    with pytest.raises(ValueError, match='15 frequency bins'):
        MODULE.first_valley('1 100\n3 20\n20 10')


def test_singleton_censoring_is_explicit_and_does_not_replace_a_failed_valley(tmp_path, monkeypatch):
    histogram = tmp_path / 'histogram.tsv'
    histogram.write_text('\n'.join(f'{index} {10000//index**2}' for index in range(1, 16)))
    commands = []

    def model_run(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index('-o') + 1])
        output.mkdir()
        (output / 'model.txt').write_text('kmercov 3.601e+00 5.902e-02 61.019')
        (output / 'summary.txt').write_text('Genome Haploid Length 340,036,397 bp 363,082,335 bp')

    monkeypatch.setattr(MODULE.subprocess, 'run', model_run)
    with pytest.raises(ValueError, match='No strict low-frequency valley'):
        MODULE.fit(histogram, Path('model.R'), Path('Rscript'), tmp_path / 'valley')
    assert commands == []
    profile = MODULE.fit(histogram, Path('model.R'), Path('Rscript'), tmp_path / 'singletons',
                         fitting_domain='exclude_singletons')
    command = commands[0]
    assert command[command.index('--typical_error') + 1] == '3'
    assert command[command.index('--num_rounds') + 1] == '1'
    assert profile['protocol']['domain_start'] == 2
    assert profile['protocol']['fitting_domain'] == 'exclude_singletons'
    assert not profile['coverage_or_assembly_accuracy_certified']


def test_reported_cutoff_is_separate_from_model_precision_and_assembly_accuracy():
    profile = MODULE.parse_model('kmercov 3.601e+00 5.902e-02 61.019',
                                 'Genome Haploid Length 340,036,397 bp 363,082,335 bp')
    assert profile['graph_min_count'] == 19
    assert profile['threshold_interval_from_coverage'] == [18, 19]
    assert profile['coverage_or_assembly_accuracy_certified'] is False


def test_finite_high_coverage_interval_can_identify_a_single_integer_cutoff():
    profile = MODULE.parse_model('kmercov 1.596e+01 1.737e-02 918.36',
                                 'Genome Haploid Length 504,268,575 bp 506,469,750 bp')
    assert profile['graph_min_count'] == 80
    assert profile['threshold_interval_from_coverage'] == [80, 80]


@pytest.mark.parametrize(('coefficient', 'sizes'), [
    ('kmercov 9.203e-02 3.696e+02 0', '1,739,132 bp Inf bp'),
    ('kmercov 9.203e-02 3.696e+02 0', '1,739,132 bp 2,000,000 bp'),
    ('kmercov 0 0 0', '1,000 bp 2,000 bp'),
])
def test_degenerate_converged_fit_does_not_yield_an_automatic_threshold(coefficient, sizes):
    with pytest.raises(ValueError):
        MODULE.parse_model(coefficient, 'Genome Haploid Length ' + sizes)
