# WOMBAT

This repository contains the code used to conduct the experiments mentioned in the paper `WOMBAT: Whitebox Oracle for Molecular Benchmarking and Attribution Testing`. 

A link to the paper will be coming soon.

## Dataset

The dataset used for validating the whiteboxes (sourced from PubChem, as described in the paper) can be found here: https://huggingface.co/datasets/dmtsh/wombat-smiles

## Repository structure 

### Artifacts

* `artifacts/explainability_method_tester/20260501_115658` contains raw data from testing explainability methods on readout r_1.
* `artifacts/explainability_method_tester/20260429_221014` contains raw data from testing explainability methods on readout r_2.

These artifacts are used in various notebooks in the repository for the purpose of performing qualitative and quantitative analysis.

### Dataset preparation

* `create_dataset_local.py` converts `ttl.gz` files downloaded from PubChem's FTP server into a CSV.

* `filter_dataset_local.py` filters the aforementioned CSV file (as described in the paper) and generates ECFP4 fingerprints for all molecules. 

* `dataset/pubchem_processed_smiles_dataset.py` contains the code used to sample from all the filtered PubChem molecules to create whitebox validation datasets (as described in the paper).

* `xai_testers/explainability_method_tester.py` then downsamples the whitebox validation datasets to evaluate XAI methods, as described in the paper.

### Whiteboxes 

* The MPNN architecture (as described in the paper) is implemented in `mpnn/mpnn_arch.py`; the input encoding is located in `mpnn/molecule_converter.py`.

* The whitebox validation code is located in the files `test.py` and `mpnn/validate_whiteboxes.py`. 

* Whiteboxes for specific patterns (as described in the paper) are localised in the following files:

    - Pattern 1: `mpnn/molecule_c_detector.py`
    - Pattern 2: `mpnn/molecule_e_detector.py`
    - Pattern 3: `mpnn/molecule_f_detector.py`
    - Pattern 4: `mpnn/molecule_g_detector.py`
    - Pattern 5: `mpnn/molecule_h_detector.py`
    - Pattern 6: `mpnn/molecule_i_detector.py`
    - Pattern 7: `mpnn/molecule_j_detector.py`
    - Pattern 8: `mpnn/molecule_k_detector.py`
    - Pattern 9: `mpnn/molecule_l_detector.py`
    - Pattern 10: `mpnn/molecule_m_detector.py`
    - Pattern 11: `mpnn/molecule_n_detector.py`
    - Pattern 12: `mpnn/molecule_o_detector.py`
    - Pattern 13: `mpnn/molecule_p_detector.py`
    - Pattern 14: `mpnn/molecule_q_detector.py`
* The file `mpnn/visualize_activations.py` contains functions that can be used to visualise internal activations in MPNNs. It highlights which elements of the latent vector had values > 0.5, essentially meaning they were activated. 
  * It uses a specific notation, where the activation of the n-th element of the latent vector in the i-th layer is denoted as `{N-th letter of the alphabet}.{i}`. For example, `F.2` would be the sixth element of the activation vector after the second MPNN layer.
* Models, along with their respective SMARTS, can be found in the file `krfp_models.py`.
* `krfp_smarts.json` contains all KRFP SMARTS; `krfp_vis.py` is a helper script to visualise them (we used it to pick acyclical motifs that seemed tractable).

### XAI evaluation

* `highlight_smarts.py` is used to derive the ground truth from SMARTS strings.

* The directory `xai_methods` contains classes implementing the `AttributionMethod` interface, returning the attributions of a given model according to a specific XAI method. 

* The directory `xai_testers` contains `explainability_method_tester.py`, which performs tests of all explainability methods (as described in the paper) and saves them (with a timestamp) to the `artifacts/explainability_method_tester` directory. It also contains `gnn_explainer_grid.py` and `pgexplainer_grid.py`, used to test more specific hyperparameters of these explainers.

### Misc

* `create_figures.ipynb` is used to show the distributions of Tanimoto and Tversky distances towards specific patterns and to generate tables for the main results.
* `gnn_explainer_grid.ipynb` and `pge_grid_results.ipynb` are used to analyse the results of GNN Explainer and PGE Explainer.
* `show_explanations.ipynb` is used to generate an example explanation for visualisation purposes in the paper. It relies on the `datavis.py` file.
* For further qualitative analysis, the notebooks `why_ig_fails.ipynb` and `why_shap_fails.ipynb` are also provided.
