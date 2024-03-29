import collections
import numpy as np

class Tokenizer(object):
    def __init__(self, lower=True, split=" ") -> None:
        self.lower = lower
        self.split = split
        self.word_index = {}
        self.word_counts = collections.OrderedDict()
    
    def fit_on_texts(self, texts):
        for text in texts:
            seq = self.text_to_word_sequence(text, lower=self.lower, split=self.split)
            
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
        
        words = self.word_counts.keys()
        self.word_index = dict(zip(words, range(1, len(words) + 1)))
    
    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))       
    
    def texts_to_sequences_generator(self, texts):
        for text in texts:
            seq = self.text_to_word_sequence(text, self.lower, self.split)
            
            vect = []
            for w in seq:
                i = self.word_index[w]
                vect.append(i)
            
            yield vect       
            
    def text_to_word_sequence(text, lower, split):
        if lower:
            text = text.lower()
            
        return text.split(split)
    
def pad_sequences(
    sequences, 
    maxlen=None, 
    dtype="int32", 
    padding="pre", 
    truncating="pre", 
    value=0.0
):
    """
    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Optional Int, maximum length of all sequences. If not provided,
            sequences will be padded to the length of the longest individual
            sequence.
        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, "pre" or "post" (optional, defaults to `"pre"`):
            pad either before or after each sequence.
        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value. (Optional, defaults to 0.)

    Returns:
        Numpy array with shape `(len(sequences), maxlen)`

    Raises:
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.unicode_
    )
    if isinstance(value, str) and dtype != object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x