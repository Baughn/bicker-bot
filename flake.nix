{
  description = "Bicker-Bot: Sibling AI bots for IRC";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        pythonEnv = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.uv

            # For IRC SSL
            pkgs.openssl

            # CUDA for sentence-transformers
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn

            # Build dependencies
            pkgs.gcc
            pkgs.pkg-config
          ];

          shellHook = ''
            # Set up CUDA paths
            export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            export LD_LIBRARY_PATH="${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:$LD_LIBRARY_PATH"

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv
            fi

            # Activate venv
            source .venv/bin/activate

            # Sync dependencies
            if [ -f pyproject.toml ]; then
              echo "Syncing dependencies..."
              uv sync --all-extras
            fi

            echo "Bicker-Bot dev shell ready!"
            echo "CUDA available at: $CUDA_PATH"
          '';
        };
      }
    );
}
