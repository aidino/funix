## Machine Learning - Funix

- [x] Language: markdown, python
- [x] Documentation: https://ngohongthai.github.io/funix/

## Convert Python notebook to HTML (with table of content)

### Install `nbconvert`

**PIP**
```shell
pip install jupyter_contrib_nbextensions
```

**Conda**
```shell
conda install -c conda-forge jupyter_contrib_nbextensions
```

### Export document with table of content (toc)

```shell
jupyter nbconvert --to html_toc FILE.ipynb
```

*Note*: Fix issues `jinja2.exceptions.TemplateNotFound: toc2`

```shell
pip install "nbconvert<6"
```

https://github.com/ipython-contrib/jupyter_contrib_nbextensions/issues/1533

*Official Documentation:* https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html