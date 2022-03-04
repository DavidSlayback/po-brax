setup(
    name="po_brax",
    version="0.0.1",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
    ],
    packages=[pkg for pkg in find_packages() if pkg.startswith('po_brax')],
    install_requires=["gym>=0.21.0", "numpy<1.22,>=1.21", "numba>=0.55.1", "brax>=0.10"],
    python_requires=">=3.8",
)