## Installation:

```
pip install git+https://github.com/jet-black/dimalib
```

## Usage example:

```python

from keras.utils.data_utils import get_file
path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read()

from dimalib.nn.solution.generator import lstm
generator = lstm.LSTMCharGenerator()
generator.fit(text, max_iter=100)

prediction = generator.generate_sample_with_seed(sample_len=100, seed="test", diversity=0.1)
print(prediction)

```
