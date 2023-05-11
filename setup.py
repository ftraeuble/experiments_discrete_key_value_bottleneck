from setuptools import setup, find_packages

setup(
    name='kv_bottleneck_experiments',
    version='1.0.0',
    packages=find_packages(include=['kv_bottleneck_experiments', 'kv_bottleneck_experiments.*']),
    author='Frederik Träuble, Nasim Rahaman',
    author_email='frederik.traeuble@tuebingen.mpg.de',
    license='MIT License',
    install_requires=[
        'wandb',
        'addict',
        'torch',
        'torchvision',
        'jupyter',
        'pyyaml',
        'ftfy',
        'regex',
        'tqdm',
        'matplotlib',
        'numpy',
        'einops',
        'clip @ git+https://github.com/openai/CLIP.git#egg=clip',
        'key_value_bottleneck @ git+https://github.com/ftraeuble/discrete_key_value_bottleneck.git#egg=key_value_bottleneck',
    ]
)
