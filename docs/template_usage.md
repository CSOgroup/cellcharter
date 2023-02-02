# Using this template

Welcome to the developer guidelines! This document is split into two parts:

1.  The [repository setup](#setting-up-the-repository). This section is relevant primarily for the repository maintainer and shows how to connect
    continuous integration services and documents initial set-up of the repository.
2.  The [contributor guide](contributing.md#contributing-guide). It contains information relevant to all developers who want to make a contribution.

## Setting up the repository

### First commit

If you are reading this, you should have just completed the repository creation with :

```bash
cruft create https://github.com/scverse/cookiecutter-scverse
```

and you should have

```
cd cellcharter
```

into the new project directory. Now that you have created a new repository locally, the first step is to push it to github. To do this, you'd have to create a **new repository** on github.
You can follow the instructions directly on [github quickstart guide][].
Since `cruft` already populated the local repository of your project with all the necessary files, we suggest to _NOT_ initialize the repository with a `README.md` file or `.gitignore`, because you might encounter git conflicts on your first push.
If you are familiar with git and knows how to handle git conflicts, you can go ahead with your preferred choice.

:::{note}
If you are looking at this document in the [cookiecutter-scverse-instance][] repository documentation, throughout this document the name of the project is `cookiecutter-scverse-instance`. Otherwise it should be replaced by your new project name: `cellcharter`.
:::

Now that your new project repository has been created on github at `https://github.com/marcovarrone/cellcharter` you can push your first commit to github.
To do this, simply follow the instructions on your github repository page or a more verbose walkthrough here:

Assuming you are in `/your/path/to/cellcharter`. Add all files and commit.

```bash
# stage all files of your new repo
git add --all
# commit
git commit -m "first commit"
```

You'll notice that the command `git commit` installed a bunch of packages and triggered their execution: those are pre-commit! To read more about what they are and what they do, you can go to the related section [Pre-commit checks](#pre-commit-checks) in this document.

:::{note}
There is a chance that `git commit -m "first commit"` fails due to the `prettier` pre-commit formatting the file `.cruft.json`. No problem, you have just experienced what pre-commit checks do in action. Just go ahead and re-add the modified file and try to commit again:

```bash
 git add -u # update all tracked file
 git commit -m "first commit"
```

:::

Now that all the files of the newly created project have been committed, go ahead with the remaining steps:

```bash
# update the `origin` of your local repo with the remote github link
git remote add origin https://github.com/marcovarrone/cellcharter.git
# rename the default branch to main
git branch -M main
# push all your files to remote
git push -u origin main
```

Your project should be now available at `https://github.com/marcovarrone/cellcharter`. While the repository at this point can be directly used, there are few remaining steps that needs to be done in order to achieve full functionality.

### Coverage tests with _Codecov_

Coverage tells what fraction of the code is "covered" by unit tests, thereby encouraging contributors to
[write tests](contributing.md#writing-tests).
To enable coverage checks, head over to [codecov][] and sign in with your GitHub account.
You'll find more information in "getting started" section of the [codecov docs][].

In the `Actions` tab of your projects' github repository, you can see that the workflows are failing due to the **Upload coverage** step. The error message in the workflow should display something like:

```
...
    Retrying 5/5 in 2s..
    {'detail': ErrorDetail(string='Could not find a repository, try using repo upload token', code='not_found')}
Error: 404 Client Error: Not Found for url:
...
```

While [codecov docs][] has a very extensive documentation on how to get started, _if_ you are using the default settings of this template we can assume that you are using [codecov][] in a github action workflow and hence you can make use of the [codecov bot][].

To set it up, simply go to the [codecov app][] page and follow the instructions to activate it for your repository.
Once the activation is completed, go back to the `Actions` tab and re-run the failing workflows.

The workflows should now succeed and you will be able to find the code coverage at this link: `https://app.codecov.io/gh/marcovarrone/cellcharter`. You might have to wait couple of minutes and the coverage of this repository should be ~60%.

If your repository is private, you will have to specify an additional token in the repository secrets. In brief, you need to:

1. Generate a Codecov Token by clicking _setup repo_ in the codecov dashboard.
    - If you have already set up codecov in the repository by following the previous steps, you can directly go to the codecov repo webpage.
2. Go to _Settings_ and copy **only** the token `_______-____-...`.
3. Go to _Settings_ of your newly created repository on GitHub.
4. Go to _Security > Secrets > Actions_.
5. Create new repository secret with name `CODECOV_TOKEN` and paste the token generated by codecov.
6. Past these additional lines in `/.github/workflows.test.yaml` under the **Upload coverage** step:
    ```bash
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
          token: ${{ secrets.CODECOV_TOKEN }}
    ```
7. Go back to github `Actions` page an re-run previously failed jobs.

### Documentation on _readthedocs_

We recommend using [readthedocs.org][] (RTD) to build and host the documentation for your project.
To enable readthedocs, head over to [their website][readthedocs.org] and sign in with your GitHub account.
On the RTD dashboard choose "Import a Project" and follow the instructions to add your repository.

-   Make sure to choose the correct name of the default branch. On GitHub, the name of the default branch should be `main` (it has
    recently changed from `master` to `main`).
-   We recommend to enable documentation builds for pull requests (PRs). This ensures that a PR doesn't introduce changes
    that break the documentation. To do so, got to `Admin -> Advanced Settings`, check the
    `Build pull requests for this projects` option, and click `Save`. For more information, please refer to
    the [official RTD documentation](https://docs.readthedocs.io/en/stable/pull-requests.html).
-   If you find the RTD builds are failing, you can disable the `fail_on_warning` option in `.readthedocs.yaml`.

If your project is private, there are ways to enable docs rendering on [readthedocs.org][] but it is more cumbersome and requires a different subscription for read the docs. See a guide [here](https://docs.readthedocs.io/en/stable/guides/importing-private-repositories.html).

### Pre-commit checks

[Pre-commit][] checks are fast programs that
check code for errors, inconsistencies and code styles, before the code
is committed.

We recommend setting up [pre-commit.ci][] to enforce consistency checks on every commit
and pull-request.

To do so, head over to [pre-commit.ci][] and click "Sign In With GitHub". Follow
the instructions to enable pre-commit.ci for your account or your organization. You
may choose to enable the service for an entire organization or on a per-repository basis.

Once authorized, pre-commit.ci should automatically be activated.

#### Overview of pre-commit hooks used by the template

The following pre-commit checks are for code style and format:

-   [black](https://black.readthedocs.io/en/stable/): standard code
    formatter in Python.
-   [isort](https://pycqa.github.io/isort/): sort module imports into
    sections and types.
-   [prettier](https://prettier.io/docs/en/index.html): standard code
    formatter for non-Python files (e.g. YAML).
-   [blacken-docs](https://github.com/asottile/blacken-docs): black on
    python code in docs.

The following pre-commit checks are for errors and inconsistencies:

-   [flake8](https://flake8.pycqa.org/en/latest/): standard check for errors in Python files.
    -   [flake8-tidy-imports](https://github.com/adamchainz/flake8-tidy-imports):
        tidy module imports.
    -   [flake8-docstrings](https://github.com/PyCQA/flake8-docstrings):
        pydocstyle extension of flake8.
    -   [flake8-rst-docstrings](https://github.com/peterjc/e8-rst-docstrings):
        extension of `flake8-docstrings` for `rst` docs.
    -   [flake8-comprehensions](https://github.com/adamchainz/e8-comprehensions):
        write better list/set/dict comprehensions.
    -   [flake8-bugbear](https://github.com/PyCQA/flake8-bugbear):
        find possible bugs and design issues in program.
    -   [flake8-blind-except](https://github.com/elijahandrews/flake8-blind-except):
        checks for blind, catch-all `except` statements.
-   [yesqa](https://github.com/asottile/yesqa):
    remove unneccesary `# noqa` comments, follows additional dependencies listed above.
-   [autoflake](https://github.com/PyCQA/autoflake):
    remove unused imports and variables.
-   [pre-commit-hooks](https://github.com/pre-commit/pre-commit-hooks): generic pre-commit hooks.
    -   **detect-private-key**: checks for the existence of private keys.
    -   **check-ast**: check whether files parse as valid python.
    -   **end-of-file-fixer**:check files end in a newline and only a newline.
    -   **mixed-line-ending**: checks mixed line ending.
    -   **trailing-whitespace**: trims trailing whitespace.
    -   **check-case-conflict**: check files that would conflict with case-insensitive file systems.
-   [pyupgrade](https://github.com/asottile/pyupgrade):
    upgrade syntax for newer versions of the language.
-   **forbid-to-commit**: Make sure that `*.rej` files cannot be commited. These files are created by the
    [automated template sync](#automated-template-sync) if there's a merge conflict and need to be addressed manually.

### How to disable or add pre-commit checks

-   To ignore lint warnigs from **flake8**, see [Ignore certain lint warnings](#how-to-ignore-certain-lint-warnings).
-   You can add or remove pre-commit checks by simply deleting relevant lines in the `.pre-commit-config.yaml` file.
    Some pre-commit checks have additional options that can be specified either in the `pyproject.toml` or tool-specific
    config files, such as `.prettierrc.yml` for **prettier** and `.flake8` for **flake8**.

### How to ignore certain lint warnings

The [pre-commit checks](#pre-commit-checks) include [flake8](https://flake8.pycqa.org/en/latest/) which checks
for errors in Python files, including stylistic errors.

In some cases it might overshoot and you may have good reasons to ignore certain warnings.

To ignore an specific error on a per-case basis, you can add a comment `# noqa` to the offending line. You can also
specify the error ID to ignore, with e.g. `# noqa: E731`. Check the [flake8 guide][] for reference.

Alternatively, you can disable certain error messages for the entire project. To do so, edit the `.flake8`
file in the root of the repository. Add one line per linting code you wish to ignore and don't forget to add a comment.

```toml
...
# line break before a binary operator -> black does not adhere to PEP8
W503
# line break occured after a binary operator -> black does not adhere to PEP8
W504
...
```

[flake8 guide]: https://flake8.pycqa.org/en/3.1.1/user/ignoring-errors.html

### API design

Scverse ecosystem packages should operate on [AnnData][] and/or [MuData][] data structures and typically use an API
as originally [introduced by scanpy][scanpy-api] with the following submodules:

-   `pp` for preprocessing
-   `tl` for tools (that, compared to `pp` generate interpretable output, often associated with a corresponding plotting
    function)
-   `pl` for plotting functions

You may add additional submodules as appropriate. While we encourage to follow a scanpy-like API for ecosystem packages,
there may also be good reasons to choose a different approach, e.g. using an object-oriented API.

[scanpy-api]: https://scanpy.readthedocs.io/en/stable/usage-principles.html

### Using VCS-based versioning

By default, the template uses hard-coded version numbers that are set in `pyproject.toml` and [managed with
bump2version](contributing.md#publishing-a-release). If you prefer to have your project automatically infer version numbers from git
tags, it is straightforward to switch to vcs-based versioning using [hatch-vcs][].

In `pyproject.toml` add the following changes, and you are good to go!

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,11 +1,11 @@
 [build-system]
 build-backend = "hatchling.build"
-requires = ["hatchling"]
+requires = ["hatchling", "hatch-vcs"]


 [project]
 name = "cellcharter"
-version = "0.3.1dev"
+dynamic = ["version"]

@@ -60,6 +60,9 @@
+[tool.hatch.version]
+source = "vcs"
+
 [tool.coverage.run]
 source = ["cellcharter"]
 omit = [
```

Don't forget to update the [Making a release section](contributing.md#publishing-a-release) in this document accordingly, after you are done!

[hatch-vcs]: https://pypi.org/project/hatch-vcs/

### Automated template sync

Automated template sync is enabled by default. This means that every night, a GitHub action runs [cruft][] to check
if a new version of the `scverse-cookiecutter` template got released. If there are any new changes, a pull request
proposing these changes is created automatically. This helps keeping the repository up-to-date with the latest
coding standards.

It may happen that a template sync results in a merge conflict. If this is the case a `*.ref` file with the
diff is created. You need to manually address these changes and remove the `.rej` file when you are done.
The pull request can only be merged after all `*.rej` files have been removed.

:::{tip}
The following hints may be useful to work with the template sync:

-   GitHub automatically disables scheduled actions if there has been not activity to the repository for 60 days.
    You can re-enable or manually trigger the sync by navigating to `Actions` -> `Sync Template` in your GitHub repository.
-   If you want to ignore certain files from the template update, you can add them to the `[tool.cruft]` section in the
    `pyproject.toml` file in the root of your repository. More details are described in the
    [cruft documentation][cruft-update-project].
-   To disable the sync entirely, simply remove the file `.github/workflows/sync.yaml`.

:::

[cruft]: https://cruft.github.io/cruft/
[cruft-update-project]: https://cruft.github.io/cruft/#updating-a-project

## Moving forward

You have reached the end of this document. Congratulations! You have successfully set up your project and are ready to start.
For everything else related to documentation, code style, testing and publishing your project ot pypi, please refer to the [contributing docs](contributing.md#contributing-guide).

<!-- Links -->

[scanpy developer guide]: https://scanpy.readthedocs.io/en/latest/dev/index.html
[cookiecutter-scverse-instance]: https://cookiecutter-scverse-instance.readthedocs.io/en/latest/template_usage.html
[github quickstart guide]: https://docs.github.com/en/get-started/quickstart/create-a-repo?tool=webui
[codecov]: https://about.codecov.io/sign-up/
[codecov docs]: https://docs.codecov.com/docs
[codecov bot]: https://docs.codecov.com/docs/team-bot
[codecov app]: https://github.com/apps/codecov
[pre-commit.ci]: https://pre-commit.ci/
[readthedocs.org]: https://readthedocs.org/
[myst-nb]: https://myst-nb.readthedocs.io/en/latest/
[jupytext]: https://jupytext.readthedocs.io/en/latest/
[pre-commit]: https://pre-commit.com/
[anndata]: https://github.com/scverse/anndata
[mudata]: https://github.com/scverse/mudata
[pytest]: https://docs.pytest.org/
[semver]: https://semver.org/
[sphinx]: https://www.sphinx-doc.org/en/master/
[myst]: https://myst-parser.readthedocs.io/en/latest/intro.html
[numpydoc-napoleon]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[sphinx autodoc typehints]: https://github.com/tox-dev/sphinx-autodoc-typehints
