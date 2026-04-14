import os
import sys

import click
import questionary
from click import pass_context

@click.command(
    "fix",
    short_help="Deprecated in Mujoco-only version."
)
@pass_context
def cli(ctx):
    questionary.print(
        "This command is deprecated. Gazebo logic removed. Only Mujoco is supported.",
        style="bold italic fg:green",
    )
    sys.exit(os.EX_OK)
