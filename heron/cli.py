import click

from heron import inference
from heron import injection
from heron import train


@click.group()
def heron():
    """
    This is the main command line program for the heron package.
    """
    pass


heron.add_command(inference.inference)
heron.add_command(inference.aspire)
heron.add_command(injection.injection)
heron.add_command(train.train)
