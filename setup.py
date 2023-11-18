import setuptools

setuptools.setup(
        name = 'pmlm',
        packages = setuptools.find_packages(),
        install_requires=[
            'numpy', 'torch', 'tqdm'],
        entry_points = {
            'console_scripts':['pmlm = pmlm.pmlm:main']})

