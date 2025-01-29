set -eu

export PYTHONUNBUFFERED=true

VIRTUUALENV=.data/venv

if [ ! -d $VIRTUUALENV ]; then
    python3 -m venv $VIRTUUALENV
    source $VIRTUUALENV/bin/activate
    pip install -r requirements.txt
else
    source $VIRTUUALENV/bin/activate
fi

if [ ! -f $VIRTUUALENV/bin/pip ]; then
    curl --silent --show-error --retry 5 https://bootstrap.pypa.io/get-pip.py | $VIRTUUALENV/bin/python
fi

$VIRTUUALENV/bin/pip install -r requirements.txt

$VIRTUUALENV/bin/python main.py