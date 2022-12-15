import click

from heron import inference

@click.group()
def heron():
    """
    This is the main command line program for the heron package.
    """
    pass

heron.add_command(inference.inference)
