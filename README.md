# Project repo for Neural Data Science 2026
## Lucía Grande González, Andre Potthoff, Niclas Collmer

To work in this repository install the required dependencies and tools by running `setup.sh`.
This sets up the correct folders initializes a git repository, installs `pre-commit` hooks, conda / python packages and configures git to use `nbdime`.


You can also do all of this manually of course.

```bash
conda env create -f environment.yml
```

or via

```bash
python3 -m pip install -r requirements.txt
```

Set up git.

```bash
git init
```

`pre-commit` hooks for working with notebooks can be installed with 

```bash
pre-commit install
```

Set up `nbdime`.
```bash
nbdime config-git --enable
```