_We use the same Contribution Guidelines as [Jaxley](https://github.com/jaxleyverse/jaxley-models/blob/main/CONTRIBUTING.md)_

## User experiences, bugs, and feature requests

To report bugs and suggest features (including better documentation), please
head over to [issues on GitHub](https://github.com/jaxleyverse/jaxley-models/issues). If you
have a question, please open
[a discussion on GitHub](https://github.com/jaxleyverse/jaxley-models/discussions).

## Code contributions

In general, we use pull requests to make changes to `Jaxley-Models`. So, if you are planning to
make a contribution, please fork, create a feature branch and then make a PR from
your feature branch to the upstream `Jaxley-Models` ([details](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)).

### Development environment

Clone [the repo](https://github.com/jaxleyverse/jaxley-models) and install
using `pip install -e ".[dev]"` (the dev flag installs development and testing
dependencies).

### Style conventions

For docstrings and comments, we use [Google
Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

Code needs to pass through the following tools, which are installed alongside `Jaxley-Models`:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can
run black manually from the console using `black .` in the top directory of the
repository, which will format all files.

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order
imports. You can run isort manually from the console using `isort` in the top
directory.

`black` and `isort` are checked as part of our CI actions. If these
checks fail please make sure you have installed the latest versions for each of them
and run them locally.