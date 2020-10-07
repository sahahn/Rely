from re import sub, template
import nibabel as nib
import numpy as np
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import os


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
    resid_data = model.intercept_ + dif

    return resid_data

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


def get_corr_size_n(resid_data, base_cohens, n, thresh=None):

    # Set a random seed based on another random seed (useful in multiproc context)
    np.random.seed(random.randint(1, 10000000))

    # Select a random group of size n from the resid_data
    index = np.arange(0, len(resid_data))
    choices = np.random.choice(index, n, replace=False)

    # Calculate the cohens on this subset
    cohens = get_cohens(resid_data[choices])

    # In the case that the calculated cohens map has any NaN
    # calculate a mask of only non NaN values in either cohens
    valid = get_non_nan_overlap_mask(cohens, base_cohens)

    if thresh is not None:
        valid = valid & (np.abs(base_cohens) > thresh)

    # Calculate the correlation only for the overlap
    corr = np.corrcoef(cohens[valid], base_cohens[valid])[0][1]

    return corr

def get_corrs(x_labels, r2, base_cohens, thresh, _print=print):

    corrs = []

    for n in x_labels:
        corr = get_corr_size_n(r2, base_cohens, n, thresh)
        _print('Corr size n=', n, '=', corr, level=2)
        corrs.append(corr)

    return corrs


def run_rely(covars_df, contrast, template_path, mask=None,
             proc_covars_func=None, thresh=None, min_size=5,
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

    _print('Perfoming group split')
    g1_subjects, g2_subjects = train_test_split(all_subjects,
                                                test_size=.5,
                                                random_state=2)
    _print('len(group1) =', len(g1_subjects), 'len(group2) =', len(g2_subjects))

     # Assign variable to the covariates per group
    c1 = covars_df.loc[g1_subjects]
    c2 = covars_df.loc[g2_subjects]

    # Apply processing seperately if requested
    if proc_covars_func is not None:
        _print('Applying proc_covars_func on each covars df seperately')
        c1 = proc_covars_func(c1)
        c2 = proc_covars_func(c2)

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

    # Residualize by group
    _print('Generate Residualized Data')
    r1, r2 = get_resid(c1, d1), get_resid(c2, d2)

    # Set r1 to be the base cohens
    _print('Generate Comparison Cohens Map')
    base_cohens = get_cohens(r1)

    # Print out base thresh info
    if thresh is not None:
        above_thresh=np.sum(np.abs(base_cohens) > thresh)
        _print(above_thresh, 'above passed pass thresh=', thresh)

    # Based on the pass params, generate different correlations
    x_labels = list(range(min_size, max_size, every))

    _print('Starting Reliability Test')

    if n_jobs == 1:
        all_corrs = []
        
        for repeat in range(n_repeats):
            _print('Starting Repeat:', repeat)

            corrs = get_corrs(x_labels, r2, base_cohens,
                              thresh, _print=_print)
            all_corrs.append(corrs)
    else:

        all_corrs = Parallel(n_jobs=n_jobs)(
            delayed(get_corrs)(x_labels, r2,
                               base_cohens,
                               thresh, _print=_print) for _ in range(n_repeats))

    all_corrs = np.array(all_corrs)
    corr_means = np.mean(all_corrs, axis=0)

    return x_labels, corr_means


