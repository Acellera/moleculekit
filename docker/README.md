# MoleculeKit container for running on cloud

## Build the container

First we need to build the container

```
docker build --tag moleculekit .
```

## Executing python scripts

Now we can execute python scripts from inside our container

```
docker run -it --rm --mount type=bind,source="$(pwd)",target=/workdir/ moleculekit python /workdir/test.py
```

Take care that anything we want in our current directory should be written to /workdir/ from inside the script
