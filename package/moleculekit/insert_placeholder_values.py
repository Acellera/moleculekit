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

# Map pypi package names to their conda equivalents
for i in range(len(deps)):
    if deps[i].startswith("msgpack"):
        deps[i] = "msgpack-python"
    elif deps[i].startswith("acellera-propka"):
        # acellera-propka is a pypi-only fork; conda uses upstream propka
        deps[i] = "propka"

# Fix conda meta.yaml
with open("package/moleculekit/recipe_template.yaml", "r") as f:
    recipe = yaml.load(f, Loader=yaml.FullLoader)

recipe["package"]["version"] = __version__
recipe["requirements"]["run"] += deps

with open("package/moleculekit/recipe.yaml", "w") as f:
    yaml.dump(recipe, f)
