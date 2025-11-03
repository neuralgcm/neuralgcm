# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# Print Python environment info for easier debugging on ReadTheDocs
import inspect
import operator
import os
import sys
import subprocess

import dinosaur  # verify this works
import neuralgcm  # verify this works

print("python exec:", sys.executable)
print("sys.path:", sys.path)
print("pip environment:")
subprocess.run([sys.executable, "-m", "pip", "list"])

print(f"dinosaur: {dinosaur.__version__}, {dinosaur.__file__}")
print(f"neuralgcm: {neuralgcm.__version__}, {neuralgcm.__file__}")

# -- Project information -----------------------------------------------------

project = 'NeuralGCM'
copyright = '2024, Google LCC'
author = 'NeuralGCM authors'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.linkcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'myst_nb',
]

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '_templates', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    # "xarray": ("https://xarray.pydata.org/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'  # https://pradyunsg.me/furo/quickstart/
# html_logo = "neuralgcm-logo.png"
html_theme_options = {
    "source_repository": "https://github.com/neuralgcm/neuralgcm/",
    "source_branch": "main",
    "source_directory": "docs/",
    "sidebar_hide_name": True,
    "light_logo": "neuralgcm-logo-light.png",
    "dark_logo": "neuralgcm-logo-dark.png",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension config

autosummary_generate = True

# https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "off"

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_typehints
autodoc_typehints = "signature"
autodoc_typehints_description_target = "documented"
autodoc_typehints_format = "short"

# Customize code links via sphinx.ext.linkcode
# Borrowed from JAX: https://github.com/google/jax/pull/20961

def linkcode_resolve(domain, info):
  if domain != 'py':
    return None
  if not info['module']:
    return None
  if not info['fullname']:
    return None
  try:
    mod = sys.modules.get(info['module'])
    obj = operator.attrgetter(info['fullname'])(mod)
    if isinstance(obj, property):
        obj = obj.fget
    while hasattr(obj, '__wrapped__'):  # decorated functions
        obj = obj.__wrapped__
    filename = inspect.getsourcefile(obj)
    source, linenum = inspect.getsourcelines(obj)
    print(f'found source code for: {info}')
  except Exception as e:
    print(f'did not find source code for: {info}: {e}')
    return None
  filename = os.path.relpath(filename, start=os.path.dirname(neuralgcm.__file__))
  lines = f"#L{linenum}-L{linenum + len(source)}" if linenum else ""
  return f"https://github.com/neuralgcm/neuralgcm/blob/main/neuralgcm/{filename}{lines}"
