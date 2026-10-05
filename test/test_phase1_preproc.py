import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StopWordsRemover, Tokenizer

spark = SparkSession.builder.appName('TestPreproc').master('local[2]').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

test_sentences = [
    ("I don't like this, it's bad and won't work. Couldn't be worse! I’m happy though.",),
    ("That's wonderful, haven't seen anything better, they're great.",),
    ("This is not good, nobody was happy, and never have I seen such a terrible thing.",),
    ("It wasn't bad, but we're not satisfied.",),
]

df = spark.createDataFrame(test_sentences, ["review"])

# 1. Contraction expansion
contractions = [
    (r"\bwon\'t\b", "will not"),
    (r"\bcan\'t\b", "cannot"),
    (r"\bcouldn\'t\b", "could not"),
    (r"\bwouldn\'t\b", "would not"),
    (r"\bshouldn\'t\b", "should not"),
    (r"\bdon\'t\b", "do not"),
    (r"\bdoesn\'t\b", "does not"),
    (r"\bdidn\'t\b", "did not"),
    (r"\bisn\'t\b", "is not"),
    (r"\bwasn\'t\b", "was not"),
    (r"\bweren\'t\b", "were not"),
    (r"\bhaven\'t\b", "have not"),
    (r"\bhasn\'t\b", "has not"),
    (r"\bhadn\'t\b", "had not"),
    (r"\bit\'s\b", "it is"),
    (r"\bthat\'s\b", "that is"),
    (r"\bi\'m\b", "i am"),
    (r"\byou\'re\b", "you are"),
    (r"\bthey\'re\b", "they are"),
    (r"\bwe\'re\b", "we are"),
    (r"\bn\'t\b", " not"),
]

c = F.regexp_replace(F.col("review"), r"[\u2019`]", "'")
c = F.lower(c)
for pat, repl in contractions:
    c = F.regexp_replace(c, pat, repl)

c = F.regexp_replace(c, r"[^a-z0-9\s]", " ")
c = F.regexp_replace(c, r"\s+", " ")
c = F.trim(c)

df_b = df.withColumn("clean_text", c)

# 2. Stop words remover with preserved negations
default_stopwords = StopWordsRemover.loadDefaultStopWords("english")
negation_words = {
    "not", "no", "never", "neither", "nor", "nothing", "nowhere", "hardly", "barely"
}
custom_stopwords = [w for w in default_stopwords if w not in negation_words]

# 3. Negation marking (Experiment C)
# e.g., "not good" -> "not_good", "never buy" -> "never_buy"
# Regex: \b(not|no|never|neither|nor|nothing|nowhere|hardly|barely|cannot)\s+([a-z0-9]+)
negation_pattern = r"\b(not|no|never|neither|nor|nothing|nowhere|hardly|barely|cannot)\s+([a-z0-9]+)"
c_marked = F.regexp_replace(c, negation_pattern, "$1_$2")
# In case of chained negations or consecutive words, we can apply once or twice
c_marked = F.regexp_replace(c_marked, r"\s+", " ")
df_c = df.withColumn("clean_text_marked", c_marked)

print("--- EXPERIMENT B (Expanded & Preserved) ---")
for r in df_b.collect():
    print(r["clean_text"])

print("\n--- EXPERIMENT C (Negation Marking) ---")
for r in df_c.collect():
    print(r["clean_text_marked"])

# Check Tokenizer + StopWordsRemover
tok = Tokenizer(inputCol="clean_text", outputCol="tokens")
remover_default = StopWordsRemover(inputCol="tokens", outputCol="filtered_default")
remover_custom = StopWordsRemover(inputCol="tokens", outputCol="filtered_custom", stopWords=custom_stopwords)

tok_df = tok.transform(df_b)
res_default = remover_default.transform(tok_df)
res_custom = remover_custom.transform(res_default)

print("\n--- STOPWORDS REMOVAL COMPARISON ---")
for r in res_custom.select("filtered_default", "filtered_custom").collect():
    print("Default:", r["filtered_default"])
    print("Custom: ", r["filtered_custom"])
    print()

spark.stop()
