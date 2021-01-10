import os

import click

from micdrop.src.preprocessing.run_preprocessing import run_preprocessing
from micdrop.src.model.train_evaluate_model import run_fit_evaluate_model
from micdrop.utils.git_utils import get_git_root


def pre_run_checks(module):
    allowed_modules = ["preprocessing", "fit_model"]

    assert (
        module in allowed_modules
    ), f"Module {module} must be one of {allowed_modules}"

    return get_git_root(os.getcwd())


@click.command()
@click.argument("run_name", nargs=1, required=True)
@click.argument("module", nargs=1, required=True)
def run(run_name, module):
    print(f"Variables passed to the run function are: {locals()}")
    run_name = run_name.lower().strip()
    module = module.lower().strip()

    git_root_loc = pre_run_checks(module)

    if module == "preprocessing":
        run_preprocessing(git_root_loc, save_external=True)

    elif module == "fit_model":
        run_fit_evaluate_model(git_root_loc, run_id=run_name, save_external=True)

    else:
        raise NotImplementedError(f"Module {module} is invalid")


if __name__ == "__main__":
    run()
