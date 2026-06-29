# if conda is installed run conda env create -f environment.yml
# else run python3 -m pip install -r requirements.txt
if command -v conda &> /dev/null
then
    conda env create -f environment.yml
else
    python3 -m pip install -r requirements.txt
fi

# initialize git
git init

# install pre-commit hooks
pre-commit install

# enable nbdime for git
nbdime config-git --enable

# create data directory
mkdir -p data

# create notebooks directory
mkdir -p notebooks