import os

coin_libs = [
    "./libCbc.0.dylib",
    "./libCbcSolver.0.dylib",
    "./libCgl.0.dylib",
    "./libClp.0.dylib",
    "./libClpSolver.0.dylib",
    "./libcoinglpk.40.dylib",
    "./libCoinUtils.0.dylib",
    "./libOsi.0.dylib",
    "./libOsiCbc.0.dylib",
    "./libOsiClp.0.dylib",
    "./libOsiCommonTest.0.dylib",
    "./libOsiGlpk.0.dylib",
    "./libcoinasl.0.dylib",
    "./libcoinmetis.0.dylib",
    "./libcoinmumps.0.dylib",
]

user_libs = [
    "/usr/lib/libbz2.1.0.dylib",
    "/usr/lib/libedit.3.dylib",
    "/usr/lib/libncurses.5.4.dylib",
    "/usr/lib/libz.1.dylib",
    "/usr/local/lib/gcc/9/libgcc_s.1.dylib",
    "/usr/local/opt/gcc/lib/gcc/9/libgfortran.5.dylib",
    "/usr/local/opt/gcc/lib/gcc/9/libquadmath.0.dylib",
]

for binary in ["cbc"] + coin_libs:
    
    for lib in coin_libs:
        libname = os.path.basename(lib)
        os.system(f"install_name_tool -change /Users/travis/build/coin-or/dist/lib/{libname} {lib} {binary}")
    
    for lib in user_libs:
        libname = os.path.basename(lib)
        os.system(f"install_name_tool -change {lib} ./{libname} {binary}")
