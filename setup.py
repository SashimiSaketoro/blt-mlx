from setuptools import find_packages, setup

setup(
    name="bytelatent",
    version="0.1.0",
    description="Byte Latent Transformer: Patches Scale Better Than Tokens (macOS MPS fork)",
    author="Meta Platforms, Inc. and affiliates.",
    url="https://github.com/SashimiSaketoro/blt-mps",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        "sentencepiece>=0.2.0",
        "tiktoken>=0.8.0",
    ],
    # xformers is optional - it fails to build on macOS
    # Install separately if needed: pip install xformers (Linux/CUDA only)
)
