
# Arabic Dialect Identification

This repo contains Machine Learning and Deep Learning approachs to classify about 500K arabic tweets into 18 dialects.


## Data fetching

Fetch the data from `AIM Technologies` API in batchs and the maximum batch size equals 1000.

Implementation in `data_fetching.ipynb` notebook.
## Preprocessing

Clean the dataset from non-arabic words and symbols. Functions are implemented in the `data_cleaning.py` script and the dataset cleaning in `dataset_cleaning.ipynb` notebook.

Preprocessing steps

- Convert `special arabic letters` into `normal letters`.

```python
special_to_arabic = {
    'ٱ': 'ا',
    'ٲ': 'ا',
    'ٶ': 'ؤ',
    'ٸ': 'ئ',
    'ٺ': 'ت',
    'ٻ': 'ي',
    .
    .
    .
    'ﻳ': 'ي',
    'ﻴ': 'ي',
    'ﻵ': 'لا',
    'ﻶ': 'لا',
    'ﻷ': 'لا',
    'ﻻ': 'لا',
    'ﻼ': 'لا'
}

# replace non-arabic to arabic chars
chars_table = {ord(char): special_to_arabic[char] for char in special_to_arabic.keys()}
```

- Remove `Tashkeel` and `Harakat`.
```python
def remove_tashkeel(text):
    tashkeel = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                              ٓ   | # Madd
                              ٔ   | # Hamza
                              ٕ   | # Hamza
                              ٙ   |
                              ٰ   |
                              ۙ   |
                              ۡ   |
                              ۢ   |
                              ۣ   |
                              ۧ   |
                             ۔   |
                         """, re.VERBOSE)
    text = re.sub(tashkeel, '', text)
    return text
```

- Using [Maha](https://github.com/TRoboto/Maha) liberary to complete the preprocessing

    - Keep arabic letters only.
    ```python
    from maha.cleaners.functions import keep
    
    tweet = keep(tweet, arabic=True, arabic_letters=True)
    ```
    - Normalize `Alef`, `Teh-Marboota` and so on ...
    ```python
    from maha.cleaners.functions import normalize

    tweet = normalize(tweet, all=True)
    ```
    - Remove `Tatweel`.
    ```python
    from maha.cleaners.functions import remove_tatweel

    # remove tatweel like ('اهــــــلا') to ('اهلا')
    tweet = remove_tatweel(tweet)
    ```
    - Reduce repeated letters.
    ```python
    from maha.cleaners.functions import reduce_repeated_substring

    # reduce repeated chars like ('ههههههه') to ('ههه')
    tweet = reduce_repeated_substring(tweet, min_repeated=4, reduce_to=3)
    ```
## Data split

- Split the data into `train` and `test`.
- We take 50K tweets for validation and save the two splits in `data` folder.
- Implementation in `data_split.ipynb` notebook.
## Machine learning approach

- Using `TF-IDF` for feature extraction.
- Using `Logistic Regression` for classification.
- Implementation in `machine_learning_model.ipynb` notebook.
## Deployment

- Using `FastAPI` to deploy our model.
- Implementation in `ml_api.py` script.
- For windows users you could run `run_ml_api.bat` file to run the API assuming conda environment name is `tf` or simply run the following command
```
uvicorn ml_api:app --reload
```
