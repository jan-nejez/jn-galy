# Excercise context

Imagine the Data Science team submitted word embeddings algorithm to encode words as real number vectors. You are now integrating these embeddings into a data processing pipeline and applying them to calculate semantic distance between phrases in English.
We will use pretrained Word2Vec vectors as a proxy for the output from our Data Science team.  (Word2Vec algorithm is a well known standard method and a stepping stone towards many modern methods in natural language processing).

# Rules
Your code must run otherwise the exercise will not be admitted

Keep your code clean. Your code will be evaluated for accuracy and style and readibility. Less on performance/efficiency.
You can use any free internet resources available to you including manuals, forums, templates, snippets and configs to complete the exercise. You are not allowed to consult with other people or AI engines
Use publicly available framework/tools of your choice to implement the exercise. Do not use licenced, proprietary software or libraries.

# Tasks
## Init pipeline
Download the pretrained set of Word2Vec vectors from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM

Load the word embeddings for first million vectors from their binary form using gensim library and then store them as flat file and then continue working with the flat file. See the snippet below:
"import gensim; from gensim.models import KeyedVectors;  wv = KeyedVectors.load_word2vec_format(location, binary=True, limit=1000000) ; wv.save_word2vec_format('vectors.csv')
"
## Process data
Calculate similarity of phrases in phrases.csv with each other:
- Assign each word in each phrase a Word2Vec embedding.
- Batch execution: Calculate L2 distance (Euclidean distance) or Cosine distance of each phrase to all other phrases and store results. Try to achieve this in a manner that is not compute or memory wasteful. Note that the whole phrase vector can be approximated by normalized sum of all the individual word tokens embeddings.
- On the fly execution: Create a function that takes any string, e.g. user-input phrase, and finds and return the closest match from phrases in phrases.csv and the distance
## Turn it into app
Structure your code into modules. Use OOP programming principles. Prepare setup.py or project.toml, prepare pip or conda environment. Initialize logging. Add some error handling and argument validations.


# Bonus points
Note: Not all bonus points are intented to be finished. Pick the ones you are confident completing within the agreed timeline. In case you find yourself wanting extra time to work on some of the bonus points, make sure to let your examiner know this is the case.

Bonus Point description
- Provide commentary and ideas how to structure and optimise the code in the future.
- Process the data in chunks and in parallel by spinning up processes in parallel or by using a distributed processing framework (including the ones limited to a single node such as Polars)
- Write some unit tests and test that could server as an entrypoint into the app (pytest preferably)
- Structure code as pipeline step execution. Separate pipeline orchestration elements from data manipulation elements. Separate data manipulation from IO. Configure IO via configuration.
- Write an custom error handling decorator - raises an custom exception if code of the method raised an error. Include the previous errors in args. Annotate your methods with it instead of having try/except explicitly in your methods.
- In Step 1  clean duplicates, outliers and stopwords from the phrases. In cases where the exact match is not found in the list of words from word2vec set, use the Levenshtein distance to find the closest similar word and use its vector instead
- Prepare a docker file building the image hosting the python app with all dependencies. If you are using a DB for data storage, have a separate image for the database and provide docker compose file to spawn the app.

# Result submission
Ideally submit a python project in a git repo with commits documenting your progress (can be hosted in cloud or sent via email - GitHub/Bitbucket).
Less preferably, share the solution as a zipped folder zipped folder.
