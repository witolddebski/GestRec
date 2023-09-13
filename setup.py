from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='GestRec',
    version='0.1.0',
    description='This library provides gesture recognition for the purpose of gesture control in applications',
    url='https://github.com/witolddebski/GestRec',
    author='Witold Debski',
    author_email='witolddebski97@gmail.com',
    license='MIT',
    packages=['gestrec'],
    install_requires=requirements,

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    include_package_data=True,
    package_data={'': ['models/*', 'test_images/*']},
)
