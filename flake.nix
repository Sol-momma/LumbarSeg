{
  description = "LumbarSeg development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
        "x86_64-linux"
      ];

      forAllSystems =
        function:
        nixpkgs.lib.genAttrs systems (
          system:
          function (
            import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            }
          )
        );
    in
    {
      devShells = forAllSystems (
        pkgs:
        let
          inherit (pkgs) lib stdenv;
        in
        {
          default = pkgs.mkShell {
            packages =
              [
                pkgs.git
                pkgs.nodejs_22
                pkgs.python311
                pkgs.uv
                pkgs.pkg-config
                pkgs.cmake
                pkgs.zlib
              ]
              ++ lib.optionals stdenv.isDarwin [
                pkgs.libiconv
              ];

            env = {
              PYTHONNOUSERSITE = "1";
              PIP_REQUIRE_VIRTUALENV = "1";
            };

            shellHook = ''
              echo "LumbarSeg dev shell"
              echo "Website: npm install && npm run dev"
              echo "Python:  python -m venv .venv && source .venv/bin/activate && pip install -r requirements-baseline.txt"
            '';
          };
        }
      );
    };
}
