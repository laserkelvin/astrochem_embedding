"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Language models for astrochemistry."""


if __name__ == "__main__":
    main(prog_name="astrochem_embedding")  # pragma: no cover
