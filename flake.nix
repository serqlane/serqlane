{
  description = "the serqlane programming language";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts/";
    nix-systems.url = "github:nix-systems/default";
  };

  outputs = inputs @ {
    self,
    flake-parts,
    nix-systems,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      debug = true;
      systems = import nix-systems;
      perSystem = {
        pkgs,
        system,
        self',
        ...
      }: let
        python = pkgs.python312;
        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        packageName = "serqlane";

        # js2py package doesn't build on 3.12
        fixedLark = (python.pkgs.lark.override {
          js2py = null;
        }).overrideAttrs {
          nativeCheckInputs = [];
          doCheck = false;
        };
      in {
        packages.${packageName} = python.pkgs.buildPythonPackage {
          src = ./.;
          pname = packageName;
          version = pyproject.tool.poetry.version;
          format = "pyproject";
          pythonImportsCheck = [packageName];
          nativeBuildInputs = [python.pkgs.poetry-core];
          propagatedBuildInputs = [fixedLark];
          nativeCheckInputs = [python.pkgs.pytestCheckHook];

          meta.mainProgram = packageName;
        };

        packages.default = self'.packages.${packageName};

        devShells.default = pkgs.mkShell {
          name = packageName;
          packages = with pkgs; [
            (poetry.withPlugins(ps: with ps; [poetry-plugin-up]))
            python
            just
            alejandra
            # TODO: check if these work with 3.12 yet (change to python.pkgs.*)
            python3.pkgs.black
            python3.pkgs.isort
            python3.pkgs.vulture
          ];
        };
      };
    };
}
