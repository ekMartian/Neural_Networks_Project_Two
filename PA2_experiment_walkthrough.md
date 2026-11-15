# PA2 experiment walkthrough

This is a study guide for interpreting the reference runs and preparing your own submission. The assignment handout is [PA2_question.pdf](PA2_question.pdf). Its Sections 17–18 describe the permitted AI assistance and student authorship requirements. The metrics below came from the reference implementation and its runs.

## What the project predicts

Each row contains 24 sensor readings from a simulated industrial machine and a label for one of six operating conditions. The network takes the readings and produces six scores called logits. The class with the highest score is its prediction.

We used the 1,800 training examples to update the weights. We used the 1,800 validation examples to measure learning and compare configurations. The course keeps its 50,000 test examples for grading.

## Data and training shared across Parts A–E

- We selected columns `x1` through `x24` in numeric order and kept `condition` as a separate integer label.
- For each sensor we subtracted the training mean and divided by the training population standard deviation. This gives all input columns a comparable scale and keeps the meaning of a standardized value consistent between training and validation.
- Every configuration trained from scratch on CPU for 120 epochs with SGD, batches of 64, zero initial biases, no momentum, and seed `314159`. A dedicated generator controlled the batch order. Models compared under identical initialization settings used identical starting parameters.
- At the end of each epoch, the program used the current weights to score each full dataset. Accuracy is the share of correct predictions. Loss measures how far the scores are from the targets using the formula for the run. Peak validation accuracy and epoch-120 validation accuracy answer different questions, so both are labelled.
- The five parts contain 19 prescribed runs. In the project folder, run any part with the corresponding letter, for example:

  ```bash
  OMP_NUM_THREADS=1 MPLBACKEND=Agg /Users/arnav/miniforge3/bin/python pa2.py --part b
  ```

  The script creates `part_b_results` in the current directory. Use `--part a`, `--part c`, `--part d`, or `--part e` for the others.

## Part A · Which hidden activation helps the network learn?

**Asked and run.** Train a `24 → 32 → 6` model using cross-entropy, Nielsen-normal initialization, learning rate `0.05`, and zero weight decay. Change only the hidden activation. The activation is the rule that transforms each of the 32 hidden values before passing it to the output layer. Identity passes it through, sigmoid and tanh squash it into bounded ranges, and ReLU sets negative values to zero.

| Activation | Validation at epoch 120 | Peak validation accuracy |
|---|---:|---:|
| Identity | 60.28% | 60.89% at epoch 26 |
| Sigmoid | 60.94% | 61.67% at epoch 112 |
| Tanh | 66.28% | 66.28% at epoch 120 |
| ReLU | 72.44% | 73.50% at epoch 86 |

**What to look at.** Identity reached 59.61% by epoch 10 and then stayed around 60%. With Identity, composing the two linear layers and their biases gives one affine mapping, so its boundaries between classes stay linear. Tanh and ReLU can form more complex boundaries and both improved validation accuracy in this comparison. Sigmoid was also nonlinear and finished close to Identity. ReLU led by epoch 10. Between epochs 86 and 120, its training accuracy climbed from 87.44% to 89.89% while validation fell from 73.50% to 72.44%. The divergence suggests some late overfitting. [Curves](part_a_results/validation_curves.png) · [Metrics](part_a_results/metrics.csv)

## Part B · Does changing the error formula affect learning?

**Asked and run.** Train two `24 → 32 → 6` ReLU models with learning rate `0.05`, zero weight decay, and identical Kaiming-normal starting weights. Cross-entropy takes the six raw scores and the integer label. Quadratic cost converts the scores to softmax probabilities, compares them to a vector with a 1 in the correct class and 0 elsewhere, and averages half the sum of squared errors per example.

| Loss | Validation at epoch 10 | Validation at epoch 120 | Peak validation accuracy |
|---|---:|---:|---:|
| Cross-entropy | 61.22% | 72.89% | 73.17% at epoch 119 |
| Quadratic cost | 34.89% | 67.17% | 67.17% at epoch 120 |

**What to look at.** Cross-entropy was already ahead at epoch 10 and finished 5.72 percentage points ahead. Quadratic cost improved throughout much of the run. One mechanism to investigate is that quadratic cost differentiates through softmax, which can shrink the gradients when probabilities become extreme. The curves show learning speed under the specified shared learning rate. The loss values have different scales, so compare the two runs using accuracy and inspect each loss curve on its own. [Accuracy](part_b_results/validation_accuracy.png) · [Cross-entropy loss](part_b_results/cross_entropy_loss.png) · [Quadratic loss](part_b_results/quadratic_loss.png) · [Metrics](part_b_results/metrics.csv)

## Part C · How much do the starting weights matter?

**Asked and run.** Use `24 → 64 → 32 → 6`, ReLU, cross-entropy, learning rate `0.03`, and zero decay. Compare normal weights with standard deviations `0.01` and `1.0` against Nielsen-normal and Kaiming-normal initialization. Nielsen uses a standard deviation of one divided by the square root of the layer's input width. Kaiming uses a scale designed for ReLU.

| Initialization | Validation loss at epoch 1 | Validation at epoch 120 | Peak validation accuracy |
|---|---:|---:|---:|
| Small normal | 1.7918 | 31.78% | 32.11% at epoch 119 |
| Large normal | 17.6493 | 50.83% | 52.22% at epoch 107 |
| Nielsen | 1.7416 | 73.61% | 74.11% at epoch 117 |
| Kaiming | 1.7238 | 71.72% | 71.83% at epoch 118 |

**What to look at.** The small weights gave validation loss close to the loss of a uniform six-class prediction after one epoch, and the model learned slowly. Tiny weights across stacked layers can leave weak signals for the earlier layers to learn from. The large weights produced especially high early loss, consistent with extreme scores and confident errors. Both layer-scaled setups did better by the end. Kaiming started ahead of Nielsen in validation accuracy after epoch 1, while Nielsen finished ahead. [Curves](part_c_results/validation_curves.png) · [Metrics](part_c_results/metrics.csv)

## Part D · Does a bigger network make better predictions?

**Asked and run.** Use ReLU, cross-entropy, Kaiming normal, learning rate `0.03`, and zero decay. Change the widths and number of hidden layers. The figures show validation accuracy across sizes and separate training and validation traces for the Large model.

| Size and hidden layers | Epoch-120 train accuracy | Epoch-120 validation accuracy | Train loss | Validation loss |
|---|---:|---:|---:|---:|
| Tiny, 8 | 66.50% | 61.78% | 0.8894 | 1.0122 |
| Small, 32 | 85.94% | 72.11% | 0.4234 | 0.7157 |
| Medium, 64 → 32 | 97.61% | 71.72% | 0.1304 | 0.9372 |
| Large, 128 → 128 | 100.00% | 72.56% | 0.0441 | 0.9432 |

**What to look at.** Tiny fit even the training data relatively poorly. Moving to Small improved both splits. The two bigger models fit training much better, but all three larger options finished within about one validation percentage point of each other. Large had the highest epoch-120 validation accuracy in this part. Small had the highest observed peak at 73.11% and the lowest epoch-120 validation loss. Large fitting every training example with worse validation loss is evidence of overfitting under this setup. [Validation accuracy](part_d_results/validation_accuracy.png) · [Large model curves](part_d_results/large_training_and_validation.png) · [Final table](part_d_results/final_metrics.csv) · [Metrics](part_d_results/metrics.csv)

## Part E · Does limiting weight size help the large model generalize?

**Asked and run.** Use the Large model from Part D and change only SGD's weight decay. Weight decay penalizes large parameters during training and applies to both the weights and biases. The reported L2 norm measures the combined size of the weight matrices. It excludes biases. The plotted loss is the predictive cross-entropy.

| Weight decay | Epoch-120 train accuracy | Epoch-120 validation accuracy | Weight L2 norm |
|---:|---:|---:|---:|
| 0 | 100.00% | 72.56% | 25.46 |
| 0.001 | 99.83% | 73.11% | 23.34 |
| 0.01 | 95.50% | **76.50%** | 12.09 |
| 0.03 | 82.83% | 72.89% | 6.30 |
| 0.1 | 31.50% | 29.94% | 1.79 |

**What to look at.** As decay increased, the final weights got smaller. At `0.01` the model still fit the training data well and correctly classified 1,377 of 1,800 validation examples. It improved final validation accuracy by 3.94 percentage points over zero decay. At `0.1` the model peaked at 60.67% at epoch 16 and subsequently lost accuracy on both splits. That pattern is consistent with excessive regularization. [Curves](part_e_results/validation_curves.png) · [Final table](part_e_results/final_metrics.csv) · [Metrics](part_e_results/metrics.csv)

## Optional model selection · Golden-section search for weight decay

Part E gave us a strong baseline at decay `0.01`. We tested five new values in the interval `[0.005, 0.02]`, using golden-section search on the logarithm of decay. This spaces probes by ratios rather than absolute increments. We reran the baseline for comparison. Each run used fresh weights, the same seed and batch order, and the same 120-epoch training setup as Part E. Only decay changed.

We used mean validation cross-entropy over epochs 111–120 to decide which side of the interval to probe next. After the first two probes, each step retained one existing probe and trained one new model. For selection at epoch 120, we required nine additional correct examples, higher mean accuracy over those ten epochs, and mean loss at least as low as the baseline. The nine-example cutoff is a practical decision rule, not a statistical significance threshold.

| Trial | Decay | Epoch-120 correct out of 1,800 | Epoch-120 validation accuracy | Mean validation loss, epochs 111–120 |
|---:|---:|---:|---:|---:|
| Baseline | 0.01000000 | 1,377 | 76.50% | 0.63481 |
| 1 | 0.00849056 | 1,366 | 75.89% | 0.64111 |
| 2 | 0.01177778 | 1,374 | 76.33% | 0.63417 |
| 3 | 0.01441793 | 1,378 | 76.56% | 0.64454 |
| 4 | 0.01039384 | 1,373 | 76.28% | 0.63357 |
| 5 | 0.00962109 | 1,370 | 76.11% | 0.63559 |

Trial 3 got 34 examples right that the baseline missed and missed 33 that the baseline got right. That is a net gain of one example, with higher mean validation loss. Trial 4 had the lowest mean loss among the tested decays and classified four fewer examples correctly at epoch 120. No candidate met the promotion rule, so the recommendation stays at `0.01`. The search provides evidence about this local interval on the supplied validation split. It cannot establish the global optimum or hidden-test accuracy. [Trial order and search intervals](decay_search_results/trials.csv) · [Epoch metrics](decay_search_results/metrics.csv) · [Recommendation](decay_search_results/recommendation.csv)

To reproduce from the project folder, run `OMP_NUM_THREADS=1 MPLBACKEND=Agg /Users/arnav/miniforge3/bin/python pa2.py --part search`.

## What is left before submission?

1. **Select the complete configuration and epoch.** The leading prescribed result is `24 → 128 → 128 → 6`, ReLU, Kaiming normal, cross-entropy, plain SGD with learning rate `0.03`, weight decay `0.01`, batch size `64`, and 120 epochs. Its 76.50% is validation accuracy. The hidden-test score and grading cluster are unknown.
2. **Interpret the six-by-six confusion matrix.** The current `0.01` candidate already has a matrix. Its rows are true validation classes and its columns are predictions. The matrix counts 1,800 predictions, with 1,377 correct. Use diagonal counts relative to row totals for per-class recall, and inspect the largest off-diagonal counts. If you choose a different decay or epoch for this architecture, rerun with `--part matrix --decay VALUE --epochs COUNT`. [Matrix](validation_results/confusion_matrix.png)
3. **Verify the final export and apply the submission filename.** The final mode trained for 120 epochs from fresh weights on the 3,600 combined rows using the original training-only means and standard deviations. The official checker accepted the AI-assisted reference export, [pa2_model.json](pa2_model.json), and a repeat run produced an identical file. The checker's 91.00% score is on rows included in final training. Use the earlier 76.50% validation result for the model-selection and generalization discussion.
4. **Write and package the submission.** The report needs the requested A–E figures, tables, interpretation, final choice, validation confusion matrix, and AI-use acknowledgment. Submit exactly `BITSID.pdf`, `BITSID.py`, and `BITSID.json` using your full capitalized BITS ID and the Moodle upload.

To reproduce the reference export with a chosen filename, run from the project folder:

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg /Users/arnav/miniforge3/bin/python pa2.py --part final --json BITSID.json
cd supporting_files
/Users/arnav/miniforge3/bin/python check_model_json.py ../BITSID.json
```

## Optional work after the prescribed runs

The handout asks you to select and justify a final model in the allowed search space. The golden-section search gives extra evidence for that choice. A further experiment, if time permits after preparing a viable checked submission, could test Nielsen initialization on the regularized Large model. It won against Kaiming on the Medium model in Part C. The assignment awards five marks for report quality, five by hidden-test performance cluster, and one for submission adherence. The course determines the clusters after evaluation.
