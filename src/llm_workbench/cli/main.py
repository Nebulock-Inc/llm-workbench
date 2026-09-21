"""The `llm-workbench` command-line interface."""

import click

from llm_workbench import __version__
from llm_workbench.cli.commands.analyze import analyze
from llm_workbench.cli.commands.calibrate import calibrate
from llm_workbench.cli.commands.dataset import dataset
from llm_workbench.cli.commands.evals import evals
from llm_workbench.cli.commands.init import init
from llm_workbench.cli.commands.models import models
from llm_workbench.cli.commands.power import power
from llm_workbench.cli.commands.run import run


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="llm-workbench")
def cli() -> None:
    """Compare LLMs on your own tasks and evals."""


cli.add_command(analyze)
cli.add_command(calibrate)
cli.add_command(dataset)
cli.add_command(evals)
cli.add_command(init)
cli.add_command(models)
cli.add_command(power)
cli.add_command(run)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
