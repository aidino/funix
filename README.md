# Machine Learning - Funix

- [x] Languages: Jupyter Notebook, Python
- [x] Doccuments page: https://ngohongthai.github.io/funix/

## Choosing the right estimator scikit-learn
![scikit learn](sklearn_map.png)

🔥 [Documentation](https://scikit-learn.org/stable/)

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