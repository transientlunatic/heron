import click

from heron import inference
from heron import injection
from heron import training_data
from heron import train_gpr


@click.group()
def heron():
    """
    This is the main command line program for the heron package.
    """
    pass


heron.add_command(inference.inference)
heron.add_command(injection.injection)
heron.add_command(training_data.training_data)
heron.add_command(train_gpr.train_gpr)
