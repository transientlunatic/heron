import click

from heron import train
from heron import evaluate as eval_module


@click.group()
def heron():
    """Heron: probabilistic waveform emulation with uncertainty."""
    pass


heron.add_command(train.train)
heron.add_command(eval_module.evaluate)
