import sys
import pathlib

project = 'GestRec'
copyright = '2023, Witold Debski'
author = 'Witold Debski'
release = '0.1'

sys.path.insert(0, pathlib.Path(__file__).parents[2].joinpath('gestrec').resolve().as_posix())
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
]
autodoc_mock_imports = ["torch", "torchvision"]
autodoc_class_signature = "separated"
master_doc = 'index'
root_doc = 'index'
autodoc_default_options = {
    'member-order': 'bysource',
    'special-members': '__init__, __call__',
    'exclude-members': '__weakref__'
}
templates_path = ['_templates']
exclude_patterns = []

html_theme = 'alabaster'
html_theme_path = []
html_static_path = ['_static']
