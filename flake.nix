{
  description = "Qwen3-TTS voice synthesis";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            python312
            ffmpeg
            libsndfile
            sox
          ];

          shellHook = ''
            export UV_PYTHON_PREFERENCE=only-system
            export UV_PYTHON=${pkgs.python312}/bin/python3
          '';
        };
      });
}
