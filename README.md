# serqlane

the serqlane programming language

## Developing
The use of [Nix](https://nixos.org/) is recommended for development. You can use this [guide](https://nixos.org/download) to install it. Once you have Nix installed and added to your PATH, you can run the following command to enter a shell with all the dependencies needed for development:  

```sh
nix develop
```
You may need to enable `nix-command` and `flakes` to use `nix develop`. Either edit your nix config or run the following command:  

``` sh
nix develop --extra-experimental-features "nix-command flakes"
```

### Inital Setup
serqlane uses [Python](https://www.python.org/) 3.12+ and [Poetry](https://python-poetry.org/) for dependency management.  

```sh
poetry install
```

### Common Commands
For running serqlane files:
```sh
poetry run serqlane <file>
```

For running tests:
```sh
poetry run pytest
```

## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) to get started with contributing to serqlane.

## License
serqlane is licensed under the [MIT License](LICENSE).