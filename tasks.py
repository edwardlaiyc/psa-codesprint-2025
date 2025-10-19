from invoke import task


@task
def format(c):
    c.run("black src cli.py gui.py src/*.py && isort src cli.py gui.py src/*.py")


@task
def install(c):
    c.run(
        "pip-compile -v --rebuild -o requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org"
    )
    c.run("pip-sync requirements.txt")
