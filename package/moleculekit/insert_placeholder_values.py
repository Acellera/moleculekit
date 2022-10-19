import toml

try:
    from moleculekit._version import __version__
except Exception:
    print("Could not get version. Defaulting to version 0")
    version = "0"

pyproject = toml.load("pyproject.toml")
deps = pyproject["project"]["dependencies"]

# Fix conda meta.yaml
with open("package/moleculekit/meta.yaml", "r") as f:
    text = f.read()

text = text.replace("BUILD_VERSION_PLACEHOLDER", __version__)

text = text.replace(
    "DEPENDENCY_PLACEHOLDER",
    "".join(["    - {}\n".format(dep.strip()) for dep in deps]),
)

with open("package/moleculekit/meta.yaml", "w") as f:
    f.write(text)
