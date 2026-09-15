# turing_mnist

A handwritten-digit classifier built entirely from scratch in [Turing](https://en.wikipedia.org/wiki/Turing_(programming_language)) — the educational language taught in Ontario high schools. No libraries, no frameworks: the forward pass, backpropagation, batching, learning-rate decay, early stopping, model serialization, and the GUI browser are all implemented by hand in ~350 lines of `main.t`. That's the point of the project.

## Architecture

A single-hidden-layer MLP:

```
784 (28×28 pixels) → 128 (sigmoid) → 10 (sigmoid)
```

- Loss: mean squared error against a one-hot target
- Optimizer: stochastic gradient descent, batch size 32
- Learning rate: 0.1, decayed as `0.1 / (1 + 0.01·epoch)`
- Early stopping: patience of 5 epochs on average epoch loss
- Weights initialized uniformly in `[-0.5, 0.5]`

## Features

- Char-by-char CSV parser for the MNIST training set (Turing has no CSV library)
- Model save/load: weights are written to `mnist_model.dat` whenever an epoch improves on the best loss
- Interactive GUI browser after training: renders each 28×28 image scaled 20×, shows the true label and the prediction (green = correct, red = wrong), navigate with the left/right arrow keys

## Running

1. Install [Open Turing](http://turing.cs.toronto.edu/) (Windows; runs under Wine on Linux).
2. Download `mnist_train.csv` from the [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) Kaggle dataset and place it next to `main.t`.
3. Open `main.t` in Open Turing and press **Run**.

The program loads the CSV, trains (or loads a saved model), then opens the image browser.

## Known issues

- **Model loading likely never triggers.** `loadModel` checks `if not modelFile > 0` after `put modelFile` — but `put modelFile` prints the stream handle to the screen rather than testing it, and `open` on a missing file doesn't produce a positive handle to check this way. In practice the program probably retrains from scratch every run. Left as-is for posterity.
- **Trains on 1,000 of 60,000 images.** `loadData` caps `imageLimit` at 1000, so accuracy is limited by the tiny training subset (and the GUI browses only those same images — there is no separate test set).
- The CSV parser reads one character at a time; loading is slow even for 1,000 rows.
- `mnistArray` is statically sized for all 60,000 rows even though only 1,000 are ever loaded.

## License

MIT — see [LICENSE](LICENSE).
