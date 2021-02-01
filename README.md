#  Pytorch NER

According to its definition on Wikipedia, Named-entity recognition (NER) (also known as entity identification, entity chunking and entity extraction) is a subtask of information extraction that seeks to locate and classify named entity mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc.


During fine-tuning, the “minimal architecture changes” required by BERT across different applications are the extra fully-connected layers. During supervised learning of a downstream application, parameters of the extra layers are learned from scratch while all the parameters in the pretrained BERT model are fine-tuned.

### Tokenization

Note that the text tokenizator can split a word into subword tokens.

```
tokenizer.encode("Driving away")
->
[0, 34002, 6645, 409, 2]
```

```
tokenizer.convert_ids_to_tokens(enc['input_ids'])
->
['<s>', 'Dri', 'ving', 'Ġaway', '</s>']
```

So we need to be careful and map each encoded token ID to its token location on the original input  
(which, seen from the point of view of the annotator, has two tokens , `['driving', 'away']`)


## Performance Evaluation

Resources:

* [NER Evalutation](http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/)

To evaluation the NER system we are measuring precision, recall and f1-score at a token level.

Comparing NER system output with the annotated dataset we can identify three different scenarios:

* Surface string and entity type match;
* System hypothesized an entity;
* System misses an entity;

Note that considering only this 3 scenarios, and discarding every other  
possible scenario we have a simple classification evaluation that can be  
measured in terms of false negatives, true positives, false negatives  
and false positives, and subsequently compute precision, recall and f1-score for each named-entity type.

But of course we are discarding partial matches, or other scenarios when the  
NER system gets the named-entity surface string correct but the type wrong,  
and we might also want to evaluate these scenarios again at a full-entity level.

* System assigns the wrong entity type;
* System gets the boundaries and entity type wrong;
* System gets the boundaries of the surface string wrong;


## Evaluation

At evaluation phase we face another problem: the model predicts tag  
annotations on the sub-word level, not on the word level.  
To obtain word-level annotations, we need to aggregate the sub-word  
level predictions for each word. Two obvious solutions come to mind:

* for each sub-word, choose the tag with highest probability, and then use a majority vote, or
* average the predicted probabilities over all sub-words of a word, and then take the tag with highest average probability.

---

See `tests/functional/test_classification.py` for details on how to use.

```bash
python src/main.py \
    --model 'roberta' \
    --output-dir 'output' \
    --data-path 'data/fake_news_sample.csv'
 ```

