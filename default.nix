{ 
    pkgs ? import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/20.09.tar.gz") {}
}:
let
    jobs = rec {
        data-shell = pkgs.mkShell {
            name = "data-shell";
            
            buildInputs = [
                (pkgs.python3.withPackages ( ps: [
                    ps.numpy
                    ps.pandas
                    ps.networkx
                ]))
            ];
        };

        jupyter-shell = pkgs.mkShell {
            name = "jupyter-shell";

            buildInputs = [
                (pkgs.python3.withPackages ( ps: [
                    ps.numpy
                    ps.pandas
                    ps.matplotlib
                    #ps.ipython
                ]))
                pkgs.jupyter
            ];
        };

        binpack = pkgs.stdenv.mkDerivation {
            name = "binpack_cpp";
            
            buildInputs = with pkgs; [
                cmake
            ];

            src = pkgs.lib.sourceByRegex ./Binpack_CPP [
                "^src"
                "^src/.*\.?pp"
                "^src/CMakeLists.txt"
                "^src/Binpack_lib"
                "^src/algos"
                "^src/Binpack_lib/.*\.?pp"
                "^src/Binpack_lib/csv.h"
                "^src/Binpack_lib/CMakeLists.txt"
                "^CMakeLists.txt"
            ];
        };
    };
in
    jobs