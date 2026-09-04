"""The `llm-lab` command-line interface."""

import click

from llm_lab import __version__
from llm_lab.cli.commands.analyze import analyze
from llm_lab.cli.commands.dataset import dataset
from llm_lab.cli.commands.evals import evals
from llm_lab.cli.commands.models import models
from llm_lab.cli.commands.power import power
from llm_lab.cli.commands.review import review
from llm_lab.cli.commands.run import run

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="llm-lab")
def cli() -> None:
    """Compare LLMs on your own tasks and evals."""


cli.add_command(analyze)
cli.add_command(dataset)
cli.add_command(evals)
cli.add_command(models)
cli.add_command(power)
cli.add_command(review)
cli.add_command(run)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
