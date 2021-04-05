import nibabel as nib
from nibabel import GiftiImage
import numpy as np
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import os
from scipy.stats import pearsonr, ttest_1samp
from statsmodels.stats.multitest import fdrcorrection

def get_resid_with_nans(covars, data):

    # Go from pandas df to numpy array
    covars = np.array(covars)

    # Make sure data is numpy array
    data = np.array(data)
    
    # Init empty resid array of NaN's
    resid = np.empty(shape=data.shape)
    resid[:] = np.nan
    
    # For each feature seperately
    for i in range(data.shape[1]):
        
        # Operate on non-nan subjects for this feature
        mask = ~np.isnan(data[:, i])
        
        # If not at least 2 subjects valid,
        # skip and propegate NaN's for this
        # voxel.
        if np.sum(mask) > 1:
        
            # Fit model
            model = LinearRegression().fit(covars[mask], data[mask, i])
            
            # Compute difference of real value - predicted
            dif_i = data[mask, i] - model.predict(covars[mask])
            
            # Set resid as diff + intercept
            resid_i = model.intercept_ + dif_i
            
            # Fill in NaN mask
            resid[mask, i] = resid_i
    
    return resid

def get_resid_base(covars, data):

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

def get_resid(covars, data):

    if np.isnan(data).any():
        return get_resid_with_nans(covars, data)
    return get_resid_base(covars, data)

def get_cohens_with_nan(data):

    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    cohen = mean / std

    return cohen

def get_cohens_base(data):

    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    cohen = mean / std

    return cohen

def get_cohens(data):

    if np.isnan(data).any():
        return get_cohens_with_nan(data)
    return get_cohens_base(data)

def fast_corr(O, P):
    
    n = P.size
    DO = O - (np.sum(O, 0) / np.double(n))
    DP = P - (np.sum(P) / np.double(n))
    return np.dot(DP, DO) / np.sqrt(np.sum(DO ** 2, 0) * np.sum(DP ** 2))

def _apply_template(subject, contrast, template_path):
    
    f = template_path
    f = f.replace('SUBJECT', subject).replace('CONTRAST', str(contrast))
    
    return f

def _raw_from_nib(data, index_slice):
    
    # for index_slice from Gifti need to do anything?
    if index_slice is not None:
        return data.dataobj[index_slice]
    
    # If Gifti
    elif isinstance(data, GiftiImage):
        return data.darrays[0].data

    # If other nibabel
    return data.get_fdata()


def _load_subject(subject, contrast, template_path, mask=None, index_slice=None, _print=print):

    # Get the specific path for this subject based on the passed template
    f = _apply_template(subject, contrast, template_path)

    _print('Loading:', f, level=2)

    # Load in the file as nibabel
    data = nib.load(f)

    # From raw to data
    raw = _raw_from_nib(data, index_slice)

    # If no brain mask, just flatten
    if mask is None:
        flat_data = raw.flatten()
    
    # Create 1D version of that cope based on the mask
    else:
        flat_data = raw[mask == 1]

    return flat_data
 
def get_data(subjects, contrast, template_path, mask=None,
             index_slice=None, n_jobs=1, _print=print):

    # Can pass mask as None, file loc, or numpy array
    if mask is not None:
        if isinstance(mask, str):
            mask = nib.load(mask).get_fdata()

    # Start a list in order to get the ordering correct
    if n_jobs == 1:

        subjs_data = []
        for subject in subjects:
            subjs_data.append(_load_subject(subject, contrast, template_path,
                                            mask=mask, index_slice=index_slice,
                                            _print=_print))

    else:
        subjs_data = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_load_subject)(
                subject=subject,
                contrast=contrast,
                template_path=template_path,
                mask=mask,
                index_slice=index_slice,
                _print=_print) for subject in subjects)

    # Return the number of subjects by the sum of the mask
    return np.stack(subjs_data)

def get_non_nan_overlap_mask(c1, c2):
    return ~np.isnan(c1) & ~np.isnan(c2)

def get_corr_size_n(covars=None, data=None, resid=None,
                    base_map=None, proc_covars_func=None,
                    perf_series=None,
                    n=None, thresh=None):

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
        
        if perf_series is None:
            ps = None
        else:
            ps = perf_series.iloc[choices].copy()
        
        compare_map = get_proc_map(covars.iloc[choices].copy(), data[choices],
                              proc_covars_func, perf_series=ps)

    # Otherwise, just calculate based on subset of passed resid
    else:

        if perf_series is None:
            compare_map = get_cohens(resid[choices])
        else:
            compare_map = fast_corr(resid[choices],
                                    np.array(perf_series.iloc[choices].copy()))

    # In the case that the calculated map has any NaN
    # calculate a mask of only non NaN values in either compare_map
    valid = get_non_nan_overlap_mask(compare_map, base_map)

    if thresh is not None:
        valid = valid & (np.abs(base_map) > thresh)

    # Calculate the correlation only for the overlap
    corr = np.corrcoef(compare_map[valid], base_map[valid])[0][1]
    corr, p_value = pearsonr(compare_map[valid], base_map[valid])

    return corr, p_value

def get_corrs(x_labels=None, covars=None, data=None, resid=None,
              base_map=None, proc_covars_func=None, perf_series=None,
              thresh=None, _print=print):

    corrs = []
    p_values = []

    for n in x_labels:
        corr, p_value =\
            get_corr_size_n(covars=covars, data=data, resid=resid, 
                            base_map=base_map, proc_covars_func=proc_covars_func,
                            perf_series=perf_series,
                            n=n, thresh=thresh)
        
        _print('Corr size n=', n, '=', corr, 'p_value=', p_value, level=2)
        corrs.append(corr)
        p_values.append(p_value)

    _print('Finished repeat!', level=1)

    return corrs, p_values

def get_proc_map(covars, data, proc_covars_func, perf_series=None):

     # Proc separate if passed
    if proc_covars_func is not None:
        covars = proc_covars_func(covars)
    
    # Residualize
    resid = get_resid(covars, data)

    # Generate cohens if no perf_series
    if perf_series is None:
        return get_cohens(resid)

    # Otherwise, generate correlation w/ perf_series
    else:
        return fast_corr(resid, np.array(perf_series))

def rely(proc_type, covars, data, base_map, proc_covars_func,
         perf_series, x_labels, n_repeats, thresh, n_jobs, _print):

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
                               base_map=base_map,
                               proc_covars_func=proc_covars_func,
                               perf_series=perf_series,
                               thresh=thresh,
                               _print=_print) for _ in range(n_repeats))

    all_corrs = [o[0] for o in output]
    all_p_values = [o[1] for o in output]
         
    return all_corrs, all_p_values

def setup_subjects(covars_df, template_path, contrast, n_jobs, verbose):

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

    missing_subjects = [s for s in covars_df.index if s not in all_subjects]
    _print('Missing:', missing_subjects)

    return all_subjects, _print

def run_rely(covars_df, contrast, template_path, mask=None,
             index_slice=None, stratify=None, proc_covars_func=None,
             perf_series=None, proc_type='split',
             thresh=None, min_size=5,
             max_size=1000, every=1, n_repeats=100,
             n_jobs=1, split_random_state=2, verbose=1):
    ''' Function for computing a basic metric of reliability.

    Notes on NaN's:
    - NaN's may not be passed in the covars_df, stratify or perf_series.

    - If there are any NaN's in the data, when residualizing each voxel/vertex,
    that subject will be excluded from calculating that voxel/vertex's residual,
    but will still be included in every case where they have a non-nan value.
    Essentially the NaN value will propegate into the residualized data.
    Likewise, when calculating the cohen's map, any nan values will simply
    be skipped when calculating the mean and std. 

    - If there are any NaN's in either the group to compares cohens map,
    or the base cohen's map, then those voxels / vertex will be excluded
    when calculating the correlation coef. between the two maps. NaN's
    will show up here if the standard deviation for that feature
    across subjects is 0.
    
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

    index_slice : slices, tuple of slices, or None
        You may optional pass index slicing here. This will be applied
        before the mask. The benefit of using index slicing over a mask
        is that in the case where say your data is of shape 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass slice(0) here, then this accomplishes the same thing,
        but only loads the requested slice from the disk.

        You must use slice for complex slicing, in the example above you could just
        pass (0) or slice(0), but if you wanted something more complex, e.g. passing
        something like my_array[1:5:2, ::3] you should pass
        (slice(1,5,2), slice(None,None,3)) here.
        
        By default this is None.

    stratify : None or pandas Series
        By default this is None. If passed a series though,
        then this series is expected to be indexed by subject,
        with the same subjects as the passed covars_df. This series
        of values will then be passed to the train_test_split function,
        and will specify that stratifying behavior is requested.

    proc_covars_func : None, function
        By default, this is set to None. Alternatively, you
        may pass a function in which the first positional argument accepts
        a subset of the covars_df, and then returns a processed version of
        the covar df. This is useful for preforming pre-processing on the
        covars_df separately for each group of subjects.

    perf_series : None or pandas Series
        By default None. If not None, then this should
        be a series indexed by subject id. The cohen's
        will not be calculated anymore, instead the reliability
        for the residualized data as correlated with this
        series will be computed instead.

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
        in the comparison map above this threshold should be used
        to calculate the correlation between the random group and this
        base map.

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

    split_random_state : int
        By default 2. Can pass different int's.
        This is the random state for the random tr test split.

    verbose : int
        By default this value is 1. This parameter
        controls the verbosity of this function.

        If -1, then no message at all will be printed.
        If 0, only warnings will be printed.
        If 1, general status updates will be printed.
        If >= 2, full verbosity will be enabled.
    '''
    
    # Get all subjects and setup
    all_subjects, _print = setup_subjects(covars_df, template_path, contrast, n_jobs, verbose)

    # Proc the stratify series
    if stratify is not None:
        stratify_vals = stratify.loc[all_subjects]
    else:
        stratify_vals = None

    _print('Performing group split, w/ stratify =', stratify is not None,
           'random_state =', split_random_state)

    # Compute the train test split on the sorted subjects (for reproducibility)
    g1_subjects, g2_subjects = train_test_split(sorted(all_subjects),
                                                test_size=.5,
                                                random_state=split_random_state,
                                                stratify=stratify_vals)

    _print('len(group1) =', len(g1_subjects), 'len(group2) =', len(g2_subjects))

    # Load the data - changes the groups if some data not found
    _print('Loading Group 1 Data')
    d1 = get_data(g1_subjects, contrast,
                  template_path, mask=mask,
                  index_slice=index_slice,
                  n_jobs=n_jobs,
                  _print=_print)

    _print('Loading Group 2 Data')
    d2  = get_data(g2_subjects, contrast,
                   template_path, mask=mask,
                   index_slice=index_slice,
                   n_jobs=n_jobs,
                   _print=_print)

     # Assign variable to the co-variates per group
    c1 = covars_df.loc[g1_subjects].copy()
    c2 = covars_df.loc[g2_subjects].copy()

    # If perf series is passed, make sure it is set to the right subjects
    if perf_series is not None:
        p1 = perf_series.loc[g1_subjects].copy()
        p2 = perf_series.loc[g2_subjects].copy()
    else:
        p1, p2 = None, None

    # Generate x_labels
    x_labels = list(range(min_size, max_size, every))

    # Get base map, cohens or perf corr
    _print('Generate Base/Comparison Map')
    base_map = get_proc_map(c1, d1, proc_covars_func, perf_series=p1)

    # Print out base thresh info, if thresh passed
    if thresh is not None:
        above_thresh=np.sum(np.abs(base_map) > thresh)
        _print(above_thresh, 'above passed pass thresh=', thresh)

    # Run rely separate based on proc_type
    all_corrs, all_p_values = rely(
        proc_type=proc_type, covars=c2, data=d2,
        base_map=base_map, proc_covars_func=proc_covars_func,
        perf_series=p2, x_labels=x_labels, n_repeats=n_repeats,
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
                    index_slice=None, resid=True, n_jobs=1, verbose=1):
    ''' Loading residualized data. Or non-residualized... 

    Note: Unlike run_rely, there is not proc_covars_func, as
    this function assumes just one group to load, therefore
    the passed covars_df should be suitably proc'ed before passing
    here.
    
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

    index_slice : slices, tuple of slices, or None
        You may optional pass index slicing here. This will be applied
        before the mask. The benefit of using index slicing over a mask
        is that in the case where say your data is of shape 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass slice(0) here, then this accomplishes the same thing,
        but only loads the requested slice from the disk.

        You must use slice for complex slicing, in the example above you could just
        pass (0) or slice(0), but if you wanted something more complex, e.g. passing
        something like my_array[1:5:2, ::3] you should pass
        (slice(1,5,2), slice(None,None,3)) here.
        
        By default this is None.

    resid : bool
        If the data should be residualized or not. True by default
        for yes, if False, then raw data will be returned.

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

    # Get all subjects and setup
    all_subjects, _print = setup_subjects(covars_df, template_path, contrast, n_jobs, verbose)

    # Set covars to just the subjects found
    covars = covars_df.loc[all_subjects].copy()

    # Load all data
    _print('Loading Data')
    data = get_data(all_subjects, contrast,
                    template_path, mask=mask,
                    index_slice=index_slice,
                    n_jobs=n_jobs,
                    _print=_print)

    if resid:

        _print('Residualizing Data')
        resid = get_resid(covars, data)

        return all_subjects, resid

    else:

        _print('Returning Raw Data')
        return all_subjects, data

def run_1samp_summary(covars_df, contrast,
                      template_path, popmean=0,
                      alpha=.05,
                      mask=None, reverse_mask=True,
                      index_slice=None,
                      n_jobs=1, verbose=1):
    '''
    For the t-test uses:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html

    For p-value correction uses:
    https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.fdrcorrection.html

    Parameters
    -----------
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

    popmean : float or array_like, optional
        Expected value in null hypothesis. If array_like,
        then it must have the same shape as a excluding the axis dimension.

        default = 0

    alpha : float
        The error rate in the fdr correction.

        covars_df : pandas DataFrame
        A pandas dataframe containing any co-variates in which
        the loaded data should be residualized by.

        Note: The dataframe must be indexed by subject name!

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

    reverse_mask : bool, optional
        Can optionally reverse the masking, assuming that no index_slice is
        passed! If mask is passed as array, then will project data
        back as array. If mask is passed as location to nii, then data will
        be put back as a nifti object.

    index_slice : slices, tuple of slices, or None
        You may optional pass index slicing here. This will be applied
        before the mask. The benefit of using index slicing over a mask
        is that in the case where say your data is of shape 5 x 20 on
        the disk, but you only need the data in the first 20, e.g.,
        so with traditional slicing that is [0], in this case by using the
        mask you have to load the full data from the disk, then slice it.
        But if you pass slice(0) here, then this accomplishes the same thing,
        but only loads the requested slice from the disk.

        You must use slice for complex slicing, in the example above you could just
        pass (0) or slice(0), but if you wanted something more complex, e.g. passing
        something like my_array[1:5:2, ::3] you should pass
        (slice(1,5,2), slice(None,None,3)) here.
        
        By default this is None.

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

    # Load resid data
    all_subjects, data =\
        load_resid_data(covars_df=covars_df, contrast=contrast, template_path=template_path, mask=mask,
                        index_slice=index_slice, resid=True, n_jobs=n_jobs, verbose=verbose)

    results = {}
    
    # Run ttest
    ttest_result = ttest_1samp(data, popmean=popmean, nan_policy='omit')
    results['statistic'] = ttest_result.statistic
    results['uncorrected_p_values'] = ttest_result.pvalue

    # Run FDR
    results['corrected_p_values'] = fdrcorrection(ttest_result.pvalue, alpha=alpha)[1]
    
    # Gen cohen's map
    results['cohens'] = get_cohens(data)
    
    # If reverse_mask, reverse
    if reverse_mask:
        for key in results:
            results[key] = reverse_mask_data(results[key], mask=mask)

    results['subjects'] = all_subjects
    return results
  
def reverse_mask_data(data, mask):
    '''If mask is None, return passed data as is.'''

    if mask is None:
        return data

    # If location to nib file
    if isinstance(mask, str):
        raw_mask = nib.load(mask)
        mask = raw_mask.get_fdata()
        mask_affine = raw_mask.affine
    
    # If already mask array
    else:
        mask_affine = None

    # If multiple subjects
    if len(data.shape) == 2:
        
        projected = []
        for subj in data:
            projected.append(_project_single_subj(subj, mask, mask_affine))

        return projected

    # If single subject
    return _project_single_subj(data, mask, mask_affine)

def _project_single_subj(subj, mask, mask_affine):

    # Make copy of mask to fill
    proj_subj = mask.copy()
    proj_subj[mask == 1] = subj
    
    # Wrap in nifti image if needed
    if mask_affine is not None:
        proj_subj = nib.Nifti1Image(proj_subj, affine=mask_affine)

    return proj_subj

