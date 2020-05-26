import os
import click

from okcupid_stackoverflow.src.bert_transformer.model_fit import run_fit_bert
from okcupid_stackoverflow.src.preprocessing.run_preprocessing import run_preprocessing
from okcupid_stackoverflow.src.tf_idf.tf_idf import run_fit_tf_idf
from okcupid_stackoverflow.utils.git_utils import get_git_root


def pre_run_checks(module):
    allowed_modules = ["pre_processing", "fit_tfidf"]

    assert (
        module in allowed_modules
    ), f"Module {module} must be one of {allowed_modules}"

    return get_git_root(os.getcwd())


@click.command()
@click.argument("run_name", nargs=1, required=True)
@click.argument("module", nargs=1, required=True)
@click.argument("use_metadata", nargs=1, required=True)
def run(run_name, module, use_metadata):
    print(f"Variables passed to the run function are: {locals()}")
    run_name = run_name.lower().strip()
    module = module.lower().strip()

    assert use_metadata.lower() in (
        "t",
        "f",
        "y",
        "n",
    ), f"Use Metadata must be in ('t', 'f', 'y', or 'n'). Provided: {use_metadata}"

    if use_metadata.lower() in ("t", "y"):
        use_metadata = True
    elif use_metadata.lower() in ("f", "n"):
        use_metadata = False

    git_root_loc = pre_run_checks(module)

    if module == "pre_processing":
        run_preprocessing(git_root_loc, save_external=True)

    elif module == "fit_tfidf":
        run_fit_tf_idf(
            git_root_loc, run_id=run_name, save_external=True, use_metadata=use_metadata
        )
    elif module == "fit_bert":
        run_fit_bert(
            git_root_loc, run_id=run_name, save_external=True, use_metadata=use_metadata
        )

    else:
        raise NotImplementedError(f"Module {module} is invalid")


if __name__ == "__main__":
    run()
