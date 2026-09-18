from pathlib import Path

from setuptools import setup
from setuptools.command.sdist import sdist


class SourceDistribution(sdist):
    def make_release_tree(self, base_dir, files):
        files = [name for name in files if not any(part.endswith(".egg-info") for part in Path(name).parts)]
        super().make_release_tree(base_dir, files)


setup(cmdclass={"sdist": SourceDistribution})
