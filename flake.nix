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

        # Pin CUDA version to match driver
        cp = pkgs.cudaPackages_12_8;

        # Build LD_LIBRARY_PATH with all required libraries
        cudaLibPath = "/run/opengl-driver/lib:" + (pkgs.lib.makeLibraryPath [
          cp.cudatoolkit
          cp.cudnn
          pkgs.stdenv.cc.cc.lib
          pkgs.libGL
          pkgs.libGLU
          pkgs.mesa
          pkgs.glib
          pkgs.zlib
          pkgs.libffi
          pkgs.openssl
        ]);

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
            cp.cudatoolkit
            cp.cudnn

            # Build dependencies
            pkgs.gcc
            pkgs.pkg-config
          ];

          shellHook = ''
            # Set up CUDA environment
            export LD_LIBRARY_PATH="${cudaLibPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export CUDA_HOME="${cp.cudatoolkit}"
            export CUDA_PATH="${cp.cudatoolkit}"
            export CUDNN_PATH="${cp.cudnn}"
            export CUDA_VISIBLE_DEVICES=0

            # Create venv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              uv venv

              source .venv/bin/activate
            else
              source .venv/bin/activate
            fi

            echo "Syncing dependencies..."
            source .venv/bin/activate

            echo "Bicker-Bot dev shell ready!"
            echo "CUDA available at: $CUDA_PATH"
          '';
        };
      }
    );
}
