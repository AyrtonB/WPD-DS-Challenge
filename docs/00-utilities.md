# Utilities



```python
#exports
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import junix
from html.parser import HTMLParser
from nbdev.export2html import convert_md

from joblib import Parallel, delayed
from scipy.stats import rankdata
from skopt import BayesSearchCV
from skopt.callbacks import check_callback

import os
import codecs
from ipypb import track
from warnings import warn
from functools import partial
from distutils.dir_util import copy_tree
from collections.abc import Iterable, Sized
from collections import defaultdict

import sklearn 
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import is_classifier, clone
from sklearn.utils.validation import indexable
from sklearn.utils import check_random_state

try:
    from sklearn.metrics import check_scoring
except ImportError:
    from sklearn.metrics.scorer import check_scoring
```

<br>

### User Inputs

```python
dev_nbs_dir = '.'
docs_dir = '../docs'
docs_nb_img_dir = f'{docs_dir}/img/nbs'
nb_img_dir = '../img/nbs'
```

<br>

### Monkey Patching `skopt`

```python
#exports
def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
                         n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                         n_points=1, iid=True, refit=True, cv=None, verbose=0,
                         pre_dispatch='2*n_jobs', random_state=None,
                         error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.fit_params = fit_params
        self.iid = None

        super(BayesSearchCV, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)
        
BayesSearchCV.__init__ = bayes_search_CV_init
```

```python
#exports
def bayes_search_CV__fit(self, X, y, groups, parameter_iterable):
    """
    Actual fitting,  performing the search over parameters.
    Taken from https://github.com/scikit-learn/scikit-learn/blob/0.18.X
                .../sklearn/model_selection/_search.py
    """
    estimator = self.estimator
    cv = sklearn.model_selection._validation.check_cv(
        self.cv, y, classifier=is_classifier(estimator))
    self.scorer_ = check_scoring(
        self.estimator, scoring=self.scoring)

    X, y, groups = indexable(X, y, groups)
    n_splits = cv.get_n_splits(X, y, groups)
    if self.verbose > 0 and isinstance(parameter_iterable, Sized):
        n_candidates = len(parameter_iterable)
        print("Fitting {0} folds for each of {1} candidates, totalling"
              " {2} fits".format(n_splits, n_candidates,
                                 n_candidates * n_splits))

    base_estimator = clone(self.estimator)
    pre_dispatch = self.pre_dispatch

    cv_iter = list(cv.split(X, y, groups))
    out = Parallel(
        n_jobs=self.n_jobs, verbose=self.verbose,
        pre_dispatch=pre_dispatch
    )(delayed(sklearn.model_selection._validation._fit_and_score)(
            clone(base_estimator),
            X, y, self.scorer_,
            train, test, self.verbose, parameters,
            fit_params=self.fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True, return_parameters=True,
            error_score=self.error_score
        )
        for parameters in parameter_iterable
        for train, test in cv_iter)

    # if one choose to see train score, "out" will contain train score info
    if self.return_train_score:
        (train_scores, test_scores, n_test_samples,
         fit_time, score_time, parameters) = zip(*out)
    else:
        from warnings import warn
        (fit_failed, test_scores, n_test_samples,
         fit_time, score_time, parameters) = zip(*[a.values() for a in out])

    candidate_params = parameters[::n_splits]
    n_candidates = len(candidate_params)

    results = dict()

    def _store(key_name, array, weights=None, splits=False, rank=False):
        """A small helper to store the scores/times to the cv_results_"""
        array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                          n_splits)
        if splits:
            for split_i in range(n_splits):
                results["split%d_%s"
                        % (split_i, key_name)] = array[:, split_i]

        array_means = np.average(array, axis=1, weights=weights)
        results['mean_%s' % key_name] = array_means
        # Weighted std is not directly available in numpy
        array_stds = np.sqrt(np.average((array -
                                         array_means[:, np.newaxis]) ** 2,
                                        axis=1, weights=weights))
        results['std_%s' % key_name] = array_stds

        if rank:
            results["rank_%s" % key_name] = np.asarray(
                rankdata(-array_means, method='min'), dtype=np.int32)

    # Computed the (weighted) mean and std for test scores alone
    # NOTE test_sample counts (weights) remain the same for all candidates n_test_samples
    n_test_samples = np.array(n_test_samples[:n_splits],
                                  dtype=np.int)

    _store('test_score', test_scores, splits=True, rank=True,
           weights=n_test_samples if self.iid else None)
    if self.return_train_score:
        _store('train_score', train_scores, splits=True)
    _store('fit_time', fit_time)
    _store('score_time', score_time)

    best_index = np.flatnonzero(results["rank_test_score"] == 1)[0]
    best_parameters = candidate_params[best_index]

    # Use one MaskedArray and mask all the places where the param is not
    # applicable for that candidate. Use defaultdict as each candidate may
    # not contain all the params
    param_results = defaultdict(partial(np.ma.array,
                                        np.empty(n_candidates,),
                                        mask=True,
                                        dtype=object))
    for cand_i, params in enumerate(candidate_params):
        for name, value in params.items():
            # An all masked empty array gets created for the key
            # `"param_%s" % name` at the first occurence of `name`.
            # Setting the value at an index also unmasks that index
            param_results["param_%s" % name][cand_i] = value

    results.update(param_results)

    # Store a list of param dicts at est_sample_counts = np.array(n_test_samples[:n_splits], key 'params'
    results['params'] = candidate_params

    self.cv_results_ = results
    self.best_index_ = best_index
    self.n_splits_ = n_splits

    if self.refit:
        # fit the best estimator using the entire dataset
        # clone first to work around broken estimators
        best_estimator = clone(base_estimator).set_params(
            **best_parameters)
        if y is not None:
            best_estimator.fit(X, y, **self.fit_params)
        else:
            best_estimator.fit(X, **self.fit_params)
        self.best_estimator_ = best_estimator
    return self

BayesSearchCV._fit = bayes_search_CV__fit
```

<br>

### Custom `sklearn` Models

We require access to the dataframe index in order to evaluate the discharge optimisation predictions accurately (as we need to group predictions by date), however, standard `sklearn` strips them and returns only numpy arrays. We'll create a custom model that preserves the index.

```python
#exports
def add_series_index(idx_arg_pos=0):
    def decorator(func):
        def decorator_wrapper(*args, **kwargs):
            input_s = args[idx_arg_pos]
            assert isinstance(input_s, (pd.Series, pd.DataFrame))
            result = pd.Series(func(*args, **kwargs), index=input_s.index)
            return result
        return decorator_wrapper
    return decorator

class PandasRandomForestRegressor(RandomForestRegressor):
    def __init__(self, n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, score_func=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, ccp_alpha=ccp_alpha, max_samples=max_samples)
    
        if score_func is None:
            self.score_func = r2_score
        else:
            self.score_func = score_func
            
    @add_series_index(1)
    def predict(self, X):
        pred = super().predict(X)
        return pred
    
    def score(self, X, y, *args, **kwargs):        
        y_pred = self.predict(X)
        score = self.score_func(y, y_pred, *args, **kwargs)
        return score
```

```python
pandas_RF = PandasRandomForestRegressor()

pandas_RF
```




    PandasRandomForestRegressor(score_func=<function r2_score at 0x000001FF6FBE6D30>)



<br>

### Converting Notebooks to Markdown

```python
#exports
def convert_file_to_json(filepath):

    with open(filepath, 'r', encoding='utf8') as f:
        contents = f.read()
        f.close()

    return json.loads(contents)

junix.exporter.convert_file_to_json = convert_file_to_json

def encode_file_as_utf8(fp):
    with codecs.open(fp, 'r') as file:
        contents = file.read(1048576)
        file.close()

        if not contents:
            pass
        else:
            with codecs.open(fp, 'w', 'utf-8') as file:
                file.write(contents)
            
def convert_nbs_to_md(nbs_dir, docs_nb_img_dir, docs_dir):
    nb_files = [f for f in os.listdir(nbs_dir) if f[-6:]=='.ipynb']

    for nb_file in track(nb_files):
        nb_fp = f'{nbs_dir}/{nb_file}'
        junix.export_images(nb_fp, docs_nb_img_dir)
        convert_md(nb_fp, docs_dir, img_path=f'{docs_nb_img_dir}/', jekyll=False)

        md_fp =  docs_dir + '/'+ nb_file.replace('.ipynb', '') + '.md'
        encode_file_as_utf8(md_fp)
```

```python
convert_nbs_to_md(dev_nbs_dir, docs_nb_img_dir, docs_dir)
```


<div><span class="Text-label" style="display:inline-block; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; min-width:0; max-width:15ex; vertical-align:middle; text-align:right"></span>
<progress style="width:60ex" max="7" value="7" class="Progress-main"/></progress>
<span class="Progress-label"><strong>100%</strong></span>
<span class="Iteration-label">7/7</span>
<span class="Time-label">[00:03<00:00, 0.37s/it]</span></div>


<br>

### Cleaning Markdown Tables

```python
#exports
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
    
    def handle_starttag(self, tag, attrs):
        self.tags.append(self.get_starttag_text())

    def handle_endtag(self, tag):
        self.tags.append(f"</{tag}>")
        
get_substring_idxs = lambda string, substring: [num for num in range(len(string)-len(substring)+1) if string[num:num+len(substring)]==substring]

def convert_df_to_md(df):
    idx_col = df.columns[0]
    df = df.set_index(idx_col)
    
    if idx_col == 'Unnamed: 0':
        df.index.name = ''
    
    table_md = df.to_markdown()
    
    return table_md

def extract_div_to_md_table(start_idx, end_idx, table_and_div_tags, file_txt):
    n_start_divs_before = table_and_div_tags[:start_idx].count('<div>')
    n_end_divs_before = table_and_div_tags[:end_idx].count('</div>')
    
    div_start_idx = get_substring_idxs(file_txt, '<div>')[n_start_divs_before-1]
    div_end_idx = get_substring_idxs(file_txt, '</div>')[n_end_divs_before]

    div_txt = file_txt[div_start_idx:div_end_idx]
    potential_dfs = pd.read_html(div_txt)
    
    assert len(potential_dfs) == 1, 'Multiple tables were found when there should be only one'
    df = potential_dfs[0]
    md_table = convert_df_to_md(df)

    return div_txt, md_table

def extract_div_to_md_tables(md_fp):
    with open(md_fp, 'r') as f:
        file_txt = f.read()
        
    parser = MyHTMLParser()
    parser.feed(file_txt)

    table_and_div_tags = [tag for tag in parser.tags if tag in ['<div>', '</div>', '<table border="1" class="dataframe">', '</table>']]
    
    table_start_tag_idxs = [i for i, tag in enumerate(table_and_div_tags) if tag=='<table border="1" class="dataframe">']
    table_end_tag_idxs = [table_start_tag_idx+table_and_div_tags[table_start_tag_idx:].index('</table>') for table_start_tag_idx in table_start_tag_idxs]

    div_to_md_tables = []

    for start_idx, end_idx in zip(table_start_tag_idxs, table_end_tag_idxs):
        div_txt, md_table = extract_div_to_md_table(start_idx, end_idx, table_and_div_tags, file_txt)
        div_to_md_tables += [(div_txt, md_table)]
        
    return div_to_md_tables

def clean_md_file_tables(md_fp):
    div_to_md_tables = extract_div_to_md_tables(md_fp)
    
    with open(md_fp, 'r') as f:
        md_file_text = f.read()

    for div_txt, md_txt in div_to_md_tables:
        md_file_text = md_file_text.replace(div_txt, md_txt)

    with open(md_fp, 'w') as f:
        f.write(md_file_text)
        
    return
```

```python
md_fps = [f'{docs_dir}/{f}' for f in os.listdir(docs_dir) if f[-3:]=='.md' if f!='00-utilities.md']

for md_fp in md_fps:
    div_to_md_tables = clean_md_file_tables(md_fp)
```

<br>

### Cleaning Image Paths

```python
#exports
def clean_md_file_img_fps(md_fp):
    with open(md_fp, 'r') as f:
        md_file_text = f.read()

    md_file_text = md_file_text.replace('../docs/img/nbs', 'img/nbs')

    with open(md_fp, 'w') as f:
        f.write(md_file_text)
        
    return
```

```python
for md_fp in md_fps:
    clean_md_file_img_fps(md_fp)
```

<br>

### Plotting

`AxTransformer` enables conversion from data coordinates to tick locations, `set_date_ticks` allows custom date ranges to be applied to plots (including a seaborn heatmap)

```python
#exports
class AxTransformer:
    def __init__(self, datetime_vals=False):
        self.datetime_vals = datetime_vals
        self.lr = linear_model.LinearRegression()
        
        return
    
    def process_tick_vals(self, tick_vals):
        if not isinstance(tick_vals, Iterable) or isinstance(tick_vals, str):
            tick_vals = [tick_vals]
            
        if self.datetime_vals == True:
            tick_vals = pd.to_datetime(tick_vals).astype(int).values
            
        tick_vals = np.array(tick_vals)
            
        return tick_vals
    
    def fit(self, ax, axis='x'):
        axis = getattr(ax, f'get_{axis}axis')()
        
        tick_locs = axis.get_ticklocs()
        tick_vals = self.process_tick_vals([label._text for label in axis.get_ticklabels()])
        
        self.lr.fit(tick_vals.reshape(-1, 1), tick_locs)
        
        return
    
    def transform(self, tick_vals):        
        tick_vals = self.process_tick_vals(tick_vals)
        tick_locs = self.lr.predict(np.array(tick_vals).reshape(-1, 1))
        
        return tick_locs
    
def set_date_ticks(ax, start_date, end_date, axis='y', date_format='%Y-%m-%d', **date_range_kwargs):
    dt_rng = pd.date_range(start_date, end_date, **date_range_kwargs)

    ax_transformer = AxTransformer(datetime_vals=True)
    ax_transformer.fit(ax, axis=axis)
    
    getattr(ax, f'set_{axis}ticks')(ax_transformer.transform(dt_rng))
    getattr(ax, f'set_{axis}ticklabels')(dt_rng.strftime(date_format))
    
    ax.tick_params(axis=axis, which='both', bottom=True, top=False, labelbottom=True)
    
    return ax
```

<br>

Finally we'll export the relevant code to our `batopt` module
