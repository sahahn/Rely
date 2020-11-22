from re import sub, template
import nibabel as nib
import numpy as np
from scipy.sparse import base
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import os
from scipy.stats import pearsonr


def get_resid(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)

    # Fit a linear regression with covars as predictors, each voxel as target variable
    model = LinearRegression().fit(covars, data)

    # The difference is the real value of the voxel, minus the predicted value
    dif = data - model.predict(covars)

    # Set the residualized data to be, the intercept of the model + the difference
    resid = model.intercept_ + dif

    return resid

def get_cohens(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    cohen = mean / std

    return cohen

def _apply_template(subject, contrast, template_path):
    
    f = template_path
    f = f.replace('SUBJECT', subject).replace('CONTRAST', str(contrast))
    
    return f

def _load_subject(subject, contrast, template_path, mask=None, _print=print):

    # Get the specific path for this subject based on the passed template
    f = _apply_template(subject, contrast, template_path)

    # Load in the file, and extract the data as np array
    _print('Loading:', f, level=2)

    data = nib.load(f)
    subject_3d_cope = data.get_fdata()

    # If no brain mask, just flatten
    if mask is None:
        flat_data = subject_3d_cope.flatten()
    
    # Create 1D version of that cope based on the mask
    else:
        flat_data = subject_3d_cope[mask == 1]

    return flat_data
 
def get_data(subjects, contrast, template_path, mask=None, n_jobs=1, _print=print):

    # Can pass mask as None, file loc, or numpy array
    if mask is not None:
        if isinstance(mask, str):
            mask = nib.load(mask).get_fdata()

    # Start a list in order to get the ordering correct
    if n_jobs == 1:

        subjs_data = []
        for subject in subjects:
            subjs_data.append(_load_subject(subject, contrast, template_path,
                                            mask=mask, _print=_print))

    else:
        subjs_data = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load_subject)(
                subject=subject,
                contrast=contrast,
                template_path=template_path,
                mask=mask,
                _print=_print) for subject in subjects)

    # Return the number of subjects by the sum of the mask
    return np.stack(subjs_data)

def get_non_nan_overlap_mask(c1, c2):
    return ~np.isnan(c1) & ~np.isnan(c2)

def get_corr_size_n(covars=None, data=None, resid=None,
                    base_cohens=None, proc_covars_func=None,
                    n=10, thresh=None):

    # Set a random seed based on another random seed (useful in multiproc context)
    np.random.seed(random.randint(1, 10000000))

    # Select a random group of size n
    if data is None and resid is not None:
        index = np.arange(0, len(resid))
    else: 
        index = np.arange(0, len(data))

    choices = np.random.choice(index, n, replace=False)

    # If split == every, go through full proc
    if data is not None and covars is not None:
        cohens = get_proc_cohens(covars.iloc[choices].copy(), data[choices], proc_covars_func)

    # Otherwise, just calculate based on subset of passed resid
    else:
        cohens = get_cohens(resid[choices])

    # In the case that the calculated cohens map has any NaN
    # calculate a mask of only non NaN values in either cohens
    valid = get_non_nan_overlap_mask(cohens, base_cohens)

    if thresh is not None:
        valid = valid & (np.abs(base_cohens) > thresh)

    # Calculate the correlation only for the overlap
    corr = np.corrcoef(cohens[valid], base_cohens[valid])[0][1]
    corr, p_value = pearsonr(cohens[valid], base_cohens[valid])

    return corr, p_value

def get_corrs(x_labels=None, covars=None, data=None, resid=None,
              base_cohens=None, proc_covars_func=None, thresh=None, _print=print):

    corrs = []
    p_values = []

    for n in x_labels:
        corr, p_value =\
            get_corr_size_n(covars=covars, data=data, resid=resid, 
                            base_cohens=base_cohens, proc_covars_func=proc_covars_func,
                            n=n, thresh=thresh)
        
        _print('Corr size n=', n, '=', corr, 'p_value=', p_value, level=2)
        corrs.append(corr)
        p_values.append(p_value)

    return corrs, p_values

def get_proc_cohens(covars, data, proc_covars_func):

    # Proc seperate if passed
    if proc_covars_func is not None:
        covars = proc_covars_func(covars)
    
    # Residualize
    resid = get_resid(covars, data)

    # Generate base cohens map
    base_cohens = get_cohens(resid)

    return base_cohens


def rely(proc_type, covars, data, base_cohens, proc_covars_func,
         x_labels, n_repeats, thresh, n_jobs, _print):

    if proc_type == 'split':
        _print('Starting Reliability Test with proc_type = "split"')
        
        # Proc seperate if passed
        if proc_covars_func is not None:
            covars = proc_covars_func(covars)
            _print('Applied proc_covars_func on each covars df seperately')

        # Residualize
        _print('Generate Residualized Data')
        resid = get_resid(covars, data)

        # Set None
        covars, data, proc_covars_func = None, None, None

    elif proc_type == 'every':
        _print('Starting Reliability Test with proc_type = "every"')

        # Set resid None
        resid = None

    else:
        raise RuntimeError('proc_type must be "split" or "every"')

    _print('Starting Reliability Test')
    output = Parallel(n_jobs=n_jobs)(
            delayed(get_corrs)(x_labels=x_labels,
                               covars=covars,
                               data=data,
                               resid=resid,
                               base_cohens=base_cohens,
                               proc_covars_func=proc_covars_func,
                               thresh=thresh,
                               _print=_print) for _ in range(n_repeats))

    all_corrs = [o[0] for o in output]
    all_p_values = [o[1] for o in output]
         
    return all_corrs, all_p_values

 
def run_rely(covars_df, contrast, template_path, mask=None,
             proc_covars_func=None, proc_type='split',
             thresh=None, min_size=5,
             max_size=1000, every=1, n_repeats=100,
             n_jobs=1, verbose=1):
    ''' Function for computing a basic metric of reliability.

    If there are any NaN's in either the group to compare,
    or the base cohens map, they will be excluded from the calculation
    of correlation between the two maps.
    
    Parameters
    ----------
    covars_df : pandas DataFrame
        A pandas dataframe containing any co-variates in which
        the loaded data should be residualized by.

        Note: The dataframe must be index'ed by subject name!

    contrast : str
        The name of the contrast, used along with the template
        path to define where to load data.

    template_path : str
        A str indicating the template form for how a single
        subjects data should be loaded, where SUBJECT will be
        replaced with that subjects name, and CONTRAST will
        be replaced with the contrast name.

        For example, so load subject X's contrast Y saved at:
        some_loc/X_Y.nii.gz.
        
        You would pass:
        some_loc/SUBJECT_CONTRAST.nii.gz

        As the template path.

    mask : str, numpy array or None
        After data is loaded, it can optionally be
        masked. By default, this parameter is set to None.
        If None, then the subjects data will be flattened.
        If passed a str, it will be assumed to be the location of a mask
        in which to load, which will then use nibabel's load function
        and then get_fdata() to extract a binary mask, where the shape
        of the mask should match the data and also entries set to True or
        1, indicate that that value be kept when loading data.
        Lastly, a numpy array, where 1 == a value should be kept, with
        likewise the same shape as the data to load can be passed here.

    proc_covars_func : None, function
        By default, this is set to None. Alternatively, you
        may pass a function in which the first positional argument accepts
        a subset of the covars_df, and then returns a processed version of
        the covar df. This is useful for preforming pre-processing on the
        covars_df seperatly for each group of subjects.

    proc_type : 'split' or 'every'
        This defines the behavior of the reliability test.
        In the first case, 'split', the covars df will be processed
        seperately across each of the two main groups (if proc_covars_func is None),
        and also each groups residualized data computed on each full half.
        
        Alternatively, if 'every' is passed, then the gold standard group will still be processed
        all together, but the comparison group will be processed and likewise
        residualized seperately for each comparison group! Warning: this
        can take longer to compute.

    thresh : float or None
        By default, this is set to None. This value if not none
        indicates an absolute value threshold in which only voxels / vertex
        in the comparison cohens map above this threshold should be used
        to calculate the correlation between the random group and this
        base cohens map.

    min_size : int
        By default this is set to 5. This is the starting
        group size to compare with the split half sample.

        Where random groups sizes are defined by:
        x_labels = list(range(min_size, max_size, every))

    max_size : int
        By default this is set to 1000. This is the maximum
        size to compare with the split half sample.

        Where random groups sizes are defined by:
        x_labels = list(range(min_size, max_size, every))

    every : int
        By default this value is 1, it determines
        jumps to the groups size.

        Where random groups sizes are defined by:
        x_labels = list(range(min_size, max_size, every))

    n_repeats : int
        By default = 100. This value controls
        how many times each of the group sizes should
        be evaluated with a different random group of that size.

    n_jobs : int
        The number of jobs to try and use for loading and the rely test.

    verbose : int
        By default this value is 1. This parameter
        controls the verbosity of this function.

        If -1, then no message at all will be printed.
        If 0, only warnings will be printed.
        If 1, general status updates will be printed.
        If >= 2, full verbosity will be enabled.
    '''

    def _print(*args, **kwargs):

        if 'level' in kwargs:
            level = kwargs.pop('level')
        else:
            level = 1

        if verbose >= level:
            print(*args, **kwargs, flush=True)

    _print('Passed covars df with shape', covars_df.shape)
    _print('Determining valid subjects')
    subj_paths = [_apply_template(s, contrast, template_path) for s in covars_df.index]
    all_subjects = Parallel(n_jobs=n_jobs, prefer="threads")(
                   delayed(os.path.exists)(s) for s in subj_paths)

    # Only include subject if found!
    all_subjects = [s for s in covars_df.index if 
                    os.path.exists(_apply_template(s, contrast, template_path))]
    _print('Found', len(all_subjects), 'subjects with data')

    _print('Perfoming group split')
    g1_subjects, g2_subjects = train_test_split(all_subjects,
                                                test_size=.5,
                                                random_state=2)
    _print('len(group1) =', len(g1_subjects), 'len(group2) =', len(g2_subjects))

    # Load the data - changes the groups if some data not found
    _print('Loading Group 1 Data')
    d1 = get_data(g1_subjects, contrast,
                  template_path, mask=mask,
                  n_jobs=n_jobs,
                  _print=_print)

    _print('Loading Group 2 Data')
    d2  = get_data(g2_subjects, contrast,
                   template_path, mask=mask,
                   n_jobs=n_jobs,
                   _print=_print)

     # Assign variable to the covariates per group
    c1 = covars_df.loc[g1_subjects].copy()
    c2 = covars_df.loc[g2_subjects].copy()

    # Generate x_labels
    x_labels = list(range(min_size, max_size, every))

    # Get base cohens
    _print('Generate Base/Comparison Cohens Map')
    base_cohens = get_proc_cohens(c1, d1, proc_covars_func)

    # Print out base thresh info, if thresh passed
    if thresh is not None:
        above_thresh=np.sum(np.abs(base_cohens) > thresh)
        _print(above_thresh, 'above passed pass thresh=', thresh)

    # Run rely seperate based on proc_type
    all_corrs, all_p_values = rely(
        proc_type=proc_type, covars=c2, data=d2,
        base_cohens=base_cohens, proc_covars_func=proc_covars_func,
        x_labels=x_labels, n_repeats=n_repeats,
        thresh=thresh, n_jobs=n_jobs, _print=_print)
  
    # Convert to array and means by repeat
    all_corrs = np.array(all_corrs)
    corr_means = np.mean(all_corrs, axis=0)
    corr_stds = np.mean(all_corrs, axis=0)

    # Same with p_values
    all_p_values = np.array(all_p_values)
    p_value_means = np.mean(all_p_values, axis=0)
    p_value_stds = np.mean(all_p_values, axis=0)

    return x_labels, corr_means, corr_stds, p_value_means, p_value_stds


def load_resid_data(covars_df, contrast, template_path, mask=None,
                    n_jobs=1, verbose=1):
    ''' Loading residualized data.
    
    Parameters
    ----------
    covars_df : pandas DataFrame
        A pandas dataframe containing any co-variates in which
        the loaded data should be residualized by.

        Note: The dataframe must be indexed by subject name!

    contrast : str
        The name of the contrast, used along with the template
        path to define where to load data.

    template_path : str
        A str indicating the template form for how a single
        subjects data should be loaded, where SUBJECT will be
        replaced with that subjects name, and CONTRAST will
        be replaced with the contrast name.

        For example, so load subject X's contrast Y saved at:
        some_loc/X_Y.nii.gz.
        
        You would pass:
        some_loc/SUBJECT_CONTRAST.nii.gz

        As the template path.

    mask : str, numpy array or None
        After data is loaded, it can optionally be
        masked. By default, this parameter is set to None.
        If None, then the subjects data will be flattened.
        If passed a str, it will be assumed to be the location of a mask
        in which to load, which will then use nibabel's load function
        and then get_fdata() to extract a binary mask, where the shape
        of the mask should match the data and also entries set to True or
        1, indicate that that value be kept when loading data.
        Lastly, a numpy array, where 1 == a value should be kept, with
        likewise the same shape as the data to load can be passed here.

    n_jobs : int
        The number of jobs to try and use for loading and the rely test.

    verbose : int
        By default this value is 1. This parameter
        controls the verbosity of this function.

        If -1, then no message at all will be printed.
        If 0, only warnings will be printed.
        If 1, general status updates will be printed.
        If >= 2, full verbosity will be enabled.
    '''

    def _print(*args, **kwargs):

        if 'level' in kwargs:
            level = kwargs.pop('level')
        else:
            level = 1

        if verbose >= level:
            print(*args, **kwargs)

    _print('Passed covars df with shape', covars_df.shape)
    _print('Determining valid subjects')
    subj_paths = [_apply_template(s, contrast, template_path) for s in covars_df.index]
    all_subjects = Parallel(n_jobs=n_jobs, prefer="threads")(
                   delayed(os.path.exists)(s) for s in subj_paths)

    # Only include subject if found!
    all_subjects = [s for s in covars_df.index if 
                    os.path.exists(_apply_template(s, contrast, template_path))]
    _print('Found', len(all_subjects), 'subjects with data')

    # Set covars to just the subjects found
    covars = covars_df.loc[all_subjects].copy()

    # Load all data
    _print('Loading Data')
    data = get_data(all_subjects, contrast,
                    template_path, mask=mask,
                    n_jobs=n_jobs,
                    _print=_print)

    _print('Residualizing Data')
    resid = get_resid(covars, data)

    return all_subjects, resid
