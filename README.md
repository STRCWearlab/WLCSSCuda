# WLCSSCuda

Cuda implementation of Warping Longest Common Subsequence. 

## Requirements

1. Nvidia CUDA enabled graphics card
2. Pycuda framework. To install it on Linux:

    ```pip install pycuda```

   or use your package manager.

## Usage

Import the main function in your code

```from wlcss__pycuda import compute_wlcss```

To use it:

```matching_scores = compute_wlcss(templates, streams, parameters)```

Where
- `templates` is a list of 1D numpy arrays containing the templates to match
- `streams` is a list of 1D numpy arrays containing the streams to match the template with
- `parameters` is a list of `[R, P, e]` parameters set for WLCSS

A matching score is computed between each template and each stream, for every parameters set.
`matching_scores` contains such scores, in a list of 1D of numpy arrays. Each array contains the scores computed
between the last sample of the template and the entire stream.

The list of scores is ordered respectively by stream, by template and then by parameter set.



