# turing_mnist

A handwritten-digit classifier built entirely from scratch in [Turing](https://en.wikipedia.org/wiki/Turing_(programming_language)) — the educational language taught in Ontario high schools. No libraries, no frameworks: the forward pass, backpropagation, learning-rate decay, early stopping, model serialization, and the GUI browser are all implemented by hand in ~360 lines of `main.t`. That's the point of the project.

## Architecture

A single-hidden-layer MLP:

```
784 (28×28 pixels) → 128 (sigmoid) → 10 (sigmoid)
```

- Loss: sum of squared errors against a one-hot target
- Optimizer: plain SGD, one weight update per sample; loss is reported every 32 samples
- Learning rate: 0.1, decayed as `0.1 / (1 + 0.01·epoch)`
- Early stopping: patience of 5 epochs on average epoch loss
- Weights initialized uniformly in `[-0.5, 0.5]`

## Features

- Char-by-char CSV parser for the MNIST training set (Turing has no CSV library)
- Model save/load: weights are written to `mnist_model.dat` whenever an epoch improves on the best loss, and loaded on the next run so training is skipped
- Interactive GUI browser after training: renders each 28×28 image scaled 20×, shows the true label and the prediction (green = correct, red = wrong), navigate with the left/right arrow keys

## Running

1. Install [Open Turing](http://turing.cs.toronto.edu/) (Windows; runs under Wine on Linux).
2. Download `mnist_train.csv` from the [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) Kaggle dataset and place it next to `main.t`.
3. Open `main.t` in Open Turing and press **Run**.

The program loads the CSV, trains (or loads a saved model), then opens the image browser.

## Known issues

- **Trains on 1,000 of 60,000 images.** `loadData` caps `IMAGE_LIMIT` at 1000, so accuracy is limited by the tiny training subset (and the GUI browses only those same images — there is no separate test set).
- The CSV parser reads one character at a time; loading is slow even for 1,000 rows.
- `loadModel` doesn't validate `mnist_model.dat`; a truncated or corrupt file will fail partway through loading. Delete it to retrain.
- The only way out of the image browser is closing the window.

## License

MIT — see [LICENSE](LICENSE).
