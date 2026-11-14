BITS F445 - Programming Assignment 2
Supporting Files
====================================

training.csv
    Training dataset for PA2.

validation.csv
    Validation dataset for model development and evaluation.

save_model_json.py
    Helper function for saving a trained PyTorch MLP in the required
    JSON format.

example_random_model.json
    Example of a correctly formatted model JSON file with randomly
    initialized parameters. It is provided only as a format example.

check_model_json.py
    Checks the submitted JSON format and computes model accuracy on
    validation.csv. Keep training.csv and validation.csv in the same
    directory when running this program.

    Usage:
        python check_model_json.py BITSID.json

Before submission, use save_model_json.py to create your final JSON file
and verify it using check_model_json.py.
