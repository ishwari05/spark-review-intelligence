"""
================================================================================
 Phase 1: Advanced Text Preprocessing Module for PySpark Sentiment Analysis
 -------------------------------------------------------------------------
 Implements:
   1. Spark-native Contraction Expansion (before tokenization/cleaning)
   2. Negation-Preserving Custom StopWords Configuration
   3. Sentiment-Preserving Text Normalization
   4. Negation-Aware Token Marking (e.g. "not good" -> "not_good")
================================================================================
"""

from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    CountVectorizer,
    IDF,
    VectorAssembler,
)

# Standard Contraction Expansion Rules (Spark Regex Patterns)
CONTRACTION_RULES = [
    # Irregular / modal negations
    (r"\bwon\'t\b", "will not"),
    (r"\bcan\'t\b", "cannot"),
    (r"\bcouldn\'t\b", "could not"),
    (r"\bwouldn\'t\b", "would not"),
    (r"\bshouldn\'t\b", "should not"),
    (r"\bmustn\'t\b", "must not"),
    (r"\bshan\'t\b", "shall not"),
    # Standard verb negations
    (r"\bdon\'t\b", "do not"),
    (r"\bdoesn\'t\b", "does not"),
    (r"\bdidn\'t\b", "did not"),
    (r"\bisn\'t\b", "is not"),
    (r"\bwasn\'t\b", "was not"),
    (r"\bweren\'t\b", "were not"),
    (r"\baren\'t\b", "are not"),
    (r"\bhaven\'t\b", "have not"),
    (r"\bhasn\'t\b", "has not"),
    (r"\bhadn\'t\b", "had not"),
    (r"\bain\'t\b", "is not"),
    # Pronoun + verb contractions
    (r"\bit\'s\b", "it is"),
    (r"\bthat\'s\b", "that is"),
    (r"\bwhat\'s\b", "what is"),
    (r"\bwhere\'s\b", "where is"),
    (r"\bhow\'s\b", "how is"),
    (r"\bthere\'s\b", "there is"),
    (r"\bhere\'s\b", "here is"),
    (r"\bwho\'s\b", "who is"),
    (r"\bi\'m\b", "i am"),
    (r"\byou\'re\b", "you are"),
    (r"\bthey\'re\b", "they are"),
    (r"\bwe\'re\b", "we are"),
    (r"\bi\'ve\b", "i have"),
    (r"\byou\'ve\b", "you have"),
    (r"\bwe\'ve\b", "we have"),
    (r"\bthey\'ve\b", "they have"),
    (r"\bi\'ll\b", "i will"),
    (r"\byou\'ll\b", "you will"),
    (r"\bhe\'ll\b", "he will"),
    (r"\bshe\'ll\b", "she will"),
    (r"\bwe\'ll\b", "we will"),
    (r"\bthey\'ll\b", "they will"),
    (r"\bi\'d\b", "i would"),
    (r"\byou\'d\b", "you would"),
    (r"\bhe\'d\b", "he would"),
    (r"\bshe\'d\b", "she would"),
    (r"\bwe\'d\b", "we would"),
    (r"\bthey\'d\b", "they would"),
    # Catch-all for any remaining n't suffixes
    (r"n\'t\b", " not"),
]

# Set of sentiment-critical negation terms to preserve from StopWords removal
PRESERVED_NEGATIONS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "nothing",
    "nowhere",
    "hardly",
    "barely",
    "nobody",
    "none",
    "without",
    "cannot",
    "cant",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "isnt",
    "wasnt",
    "werent",
    "shouldnt",
    "havent",
    "hasnt",
    "hadnt",
}


def get_sentiment_stopwords():
    """
    Returns custom stop words list excluding sentiment-critical negation words.
    """
    default_stops = StopWordsRemover.loadDefaultStopWords("english")
    return [w for w in default_stops if w not in PRESERVED_NEGATIONS]


def preprocess_variant_a(df, text_col="review", output_col="clean_text"):
    """
    EXPERIMENT A: Current existing baseline preprocessing.
    - Standard lowercasing
    - Stripping non-alphanumeric punctuation
    - Whitespace collapsing
    """
    cleaned = F.regexp_replace(F.lower(F.col(text_col)), r"[^a-z0-9\s]", " ")
    cleaned = F.regexp_replace(cleaned, r"\s+", " ")
    cleaned = F.trim(cleaned)
    return df.withColumn(output_col, cleaned)


def preprocess_variant_b(df, text_col="review", output_col="clean_text"):
    """
    EXPERIMENT B: Contraction expansion + Sentiment-preserving cleaning.
    - Normalize typographic apostrophes
    - Spark-native regex contraction expansion
    - Lowercasing and careful punctuation stripping
    - Whitespace normalization
    """
    c = F.regexp_replace(F.col(text_col), r"[\u2019\u2018\u0060\u00B4]", "'")
    c = F.lower(c)
    for pat, repl in CONTRACTION_RULES:
        c = F.regexp_replace(c, pat, repl)

    c = F.regexp_replace(c, r"[^a-z0-9\s]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    c = F.trim(c)
    return df.withColumn(output_col, c)


def preprocess_variant_c(df, text_col="review", output_col="clean_text"):
    """
    EXPERIMENT C: Contraction expansion + Sentiment-preserving cleaning + Negation Marking.
    - All steps from Variant B
    - Negation phrase marking: "not good" -> "not_good", "never buy" -> "never_buy"
    """
    # First apply variant B transformations
    c = F.regexp_replace(F.col(text_col), r"[\u2019\u2018\u0060\u00B4]", "'")
    c = F.lower(c)
    for pat, repl in CONTRACTION_RULES:
        c = F.regexp_replace(c, pat, repl)

    c = F.regexp_replace(c, r"[^a-z0-9\s]", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    c = F.trim(c)

    # Negation-aware marking: pair negation word with following token
    negation_pattern = r"\b(not|no|never|neither|nor|nothing|nowhere|hardly|barely|cannot)\s+([a-z0-9]+)"
    c = F.regexp_replace(c, negation_pattern, "$1_$2")
    c = F.regexp_replace(c, r"\s+", " ")
    c = F.trim(c)

    return df.withColumn(output_col, c)


def build_feature_stages(stop_words=None, input_col="clean_text", vocab_size=20000, min_df=2.0):
    """
    Constructs the Unigram + Bigram TF-IDF feature pipeline stages.
    If stop_words is None, default Spark stop words are used.
    If stop_words is provided, the custom stop words list is used.
    """
    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    if stop_words is not None:
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    else:
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

    ngram = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
    unigram_count_vec = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="unigram_raw_features",
        vocabSize=vocab_size,
        minDF=min_df,
    )
    unigram_idf = IDF(inputCol="unigram_raw_features", outputCol="unigram_features")
    bigram_count_vec = CountVectorizer(
        inputCol="bigrams",
        outputCol="bigram_raw_features",
        vocabSize=vocab_size,
        minDF=min_df,
    )
    bigram_idf = IDF(inputCol="bigram_raw_features", outputCol="bigram_features")
    assembler = VectorAssembler(
        inputCols=["unigram_features", "bigram_features"], outputCol="features"
    )
    return [
        tokenizer,
        remover,
        ngram,
        unigram_count_vec,
        unigram_idf,
        bigram_count_vec,
        bigram_idf,
        assembler,
    ]


def build_unigram_feature_stages(stop_words=None, input_col="clean_text", vocab_size=20000, min_df=2.0):
    """
    Constructs the Unigram-only TF-IDF feature pipeline stages.
    """
    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    if stop_words is not None:
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    else:
        remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

    unigram_count_vec = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        vocabSize=vocab_size,
        minDF=min_df,
    )
    unigram_idf = IDF(inputCol="raw_features", outputCol="features")
    return [tokenizer, remover, unigram_count_vec, unigram_idf]
