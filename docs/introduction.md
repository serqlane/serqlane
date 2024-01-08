# Introduction
This page should get you started with serqlane. The preferred method of development is using a [nix](https://nixos.org/nix/) shell, but it is not required.


## Prerequisites
If you are using [nix](https://nixos.org/nix/), it is the only prerequisite.  

Activate your shell by running `nix develop` in the root of the repository.
!!! note
    You may need to enable some experimental features if you have not done so already.  
    This can be done either by running `nix-shell --experimental-features 'nix-command flakes'` or by adding the following to your nix config:
    ```
    experimental-features = ["nix-command" "flakes"];
    ```

If you are not using nix, here is what you should have installed:  

- [Python 3.12+](https://www.python.org/downloads/)
- [Poetry](https://python-poetry.org/docs/#installation)

Some optional quality of life packages:

- [black](https://github.com/psf/black)
- [isort](https://github.com/PyCQA/isort)
- [vulture](https://github.com/jendrikseipp/vulture)

## Installation
After installing poetry you can run the following command to install all dependencies.
```sh
poetry install
```

## Running
### Serqlane Files
To run a serqlane file, run the following command:
```sh
poetry run serqlane <path-to-file>
```
### Tests
To run the tests, run the following command:
```sh
poetry run pytest
```
## Hello World
Create a file called `hello.sq` with the following contents:
```rust linenums="1"
dbg("Hello World!")
```
```
DBG: Hello World
```

## Next Steps
Now that you have serqlane installed, you can check out the [Writing Serqlane](writing-serqlane.md) to learn more about the language.