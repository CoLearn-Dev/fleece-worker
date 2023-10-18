from setuptools import setup

setup(
    name="fleece-worker",
    version="0.0.1",
    description="fleece-worker",
    author="stneng",
    author_email="git@stneng.com",
    url="https://github.com/CoLearn-Dev/fleece-worker",
    packages=["fleece-worker"],
    install_requires=[
        "numpy",
        "torch",
        "fire",
        "sentencepiece",
        "fastapi",
        "uvicorn",
        "requests",
    ],
    python_requires=">=3.10",
)
