import toml
import yaml
import setuptools_scm

try:
    __version__ = setuptools_scm.get_version()
except Exception:
    print("Could not get version. Defaulting to version 0")
    __version__ = "0"

pyproject = toml.load("pyproject.toml")
deps = pyproject["project"]["dependencies"]

# Fix msgpack pypi package which exists as msgpack-python in conda
for i in range(len(deps)):
    if deps[i].startswith("msgpack"):
        deps[i] = "msgpack-python"

# Fix conda meta.yaml
with open("package/moleculekit/recipe_template.yaml", "r") as f:
    recipe = yaml.load(f, Loader=yaml.FullLoader)

recipe["package"]["version"] = __version__
recipe["requirements"]["run"] += deps

with open("package/moleculekit/recipe.yaml", "w") as f:
    yaml.dump(recipe, f)
