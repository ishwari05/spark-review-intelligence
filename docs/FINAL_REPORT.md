# Final Project Report: Product Review Intelligence with Apache Spark

## Executive Summary

This project builds an end-to-end product-review analysis workflow using Apache Spark, Spark MLlib, and Flask. It trains and compares three sentiment classifiers, applies aspect-based sentiment analysis (ABSA), mines recurring negative-review phrases, explores latent topics, and presents the generated results in a web dashboard. In the verified run, Logistic Regression achieved the highest F1 score (0.7661) and was selected as the final model. The saved Spark pipeline and generated results are served through the Flask API.

## Objectives

- Process and analyze a large sample of labeled product reviews with Spark.
- Compare sentiment-classification models using standard evaluation metrics.
- Reuse the selected model for aspect-level and live review sentiment scoring.
- Identify recurring complaint phrases and explore themes in negative reviews.
- Expose the generated analysis and live prediction through an interactive dashboard and API.

## Dataset and Preparation

The pipeline streams a 200,000-review sample from the Amazon Polarity dataset using Hugging Face `datasets`. After cleaning and deduplication, 168,676 reviews remained; 31,324 rows were removed. The final data contains 86,008 positive reviews (50.99%) and 82,668 negative reviews (49.01%). The pipeline reports eight Spark partitions and uses an 80/20 train-test split with random seed 42.

Preprocessing removes missing, duplicate, and empty reviews; lowercases text; removes non-alphanumeric characters; normalizes whitespace; and calculates review-length features. The reported average review length is 5.24 words.

## Modeling Approach

Three Spark MLlib classifiers were evaluated:

- Naive Bayes using unigram TF-IDF features.
- Logistic Regression using unigram and bigram TF-IDF features.
- Random Forest using unigram and bigram TF-IDF features.

The model with the highest measured F1 score was selected. Logistic Regression was saved as a Spark `PipelineModel` and reused by the API for live predictions and ABSA context scoring.

## Benchmark Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0.7644 | 0.7645 | 0.7644 | 0.7642 | **0.8377** | 0.75 s |
| Logistic Regression | **0.7660** | **0.7662** | **0.7660** | **0.7661** | 0.8347 | 4.79 s |
| Random Forest | 0.6494 | 0.7400 | 0.6494 | 0.6083 | 0.7792 | 68.05 s |

Logistic Regression was selected because it had the highest F1 score and accuracy. Naive Bayes had the highest ROC-AUC, so the preferred model depends on which metric matters most to the use case.

A separate scaling demonstration recorded Logistic Regression training times of 2.04 seconds for 10,000 rows and 5.74 seconds for 200,000 rows. These are measurements from a local run and should not be treated as hardware-independent performance guarantees.

## Error Analysis

On the 33,410-review test set, the selected model made 25,593 correct predictions and 7,817 incorrect predictions, for an error rate of 23.40%. There were 3,761 false positives and 4,056 false negatives.

Negation appeared in 10.38% of the test reviews but in 15.34% of the misclassified reviews. The measured error rate among reviews containing negation was 34.56%. This indicates that short or context-poor reviews and negation remain important sources of model error. The results support further testing of negation-aware features and more diverse contextual representations.

## Aspect-Based Sentiment Analysis

The ABSA stage identified 11,301 aspect mentions across five configured aspect categories:

| Aspect | Mentions | Negative feedback |
|---|---:|---:|
| Price / Value | 4,833 | 61.66% |
| Build Quality | 2,531 | 58.55% |
| Customer Support | 1,681 | 58.54% |
| Delivery / Packaging | 1,477 | 58.36% |
| Battery | 779 | 53.66% |

All five categories were flagged as potential pain points by the pipeline's negative-feedback threshold. Price / Value had the highest negative rate and the largest mention volume. These figures are model- and lexicon-derived; keyword matching and clause-level context extraction can misattribute mentions, so the results should guide investigation rather than be interpreted as audited customer-research measurements.

## Complaint Mining and Topics

The complaint-mining stage returns 25 ranked complaint phrases. The most frequent phrase was “waste money,” with 568 occurrences, associated with Price / Value. Other frequent phrases included “poor quality” (83 occurrences) and “stopped working” (72 occurrences). Some mined phrases reflect books, films, music, and other categories in the Amazon Polarity corpus, not solely physical product defects.

Spark MLlib LDA produced five exploratory topics. Their top terms include `movie`, `book`, and `film` in one topic; `album`, `dvd`, and `music` in another; and `diapers`, `pampers`, and `baby` in another. LDA topics are statistical groupings: the terms require human interpretation and do not by themselves establish a specific complaint cause.

## Application and Delivery

The Flask application serves the static dashboard and exposes endpoints for consolidated results, aspects, complaints, topics, and live sentiment prediction. The dashboard visualizes the generated dataset summary, model benchmarks, confusion matrices, aspect analytics, complaint phrases, topics, business insights, and error examples. The live analyzer sends new reviews to the saved Spark model and returns sentiment and aspect-level results.

The verified dashboard run served the canonical `results/results.json` payload successfully. The dashboard's hero statistics were aligned to the IDs present in the HTML and verified in the browser. The current artifacts are in `results/` and the trained inference pipeline is in `saved_model/`.

## Limitations

- Spark runs locally with `local[*]`; the benchmark does not measure a distributed multi-node deployment.
- Results are based on a streamed sample of Amazon Polarity reviews, whose domains include books, films, music, household goods, and other products.
- The classifier predicts binary polarity and may fail on short, ambiguous, sarcastic, or negation-heavy reviews.
- ABSA depends on curated aspect dictionaries and clause segmentation; false matches and context errors are possible.
- Complaint phrases and LDA topics are exploratory outputs and require human review before operational decisions.
- Recorded training times are specific to this machine and run configuration.

## Running the Project

With Python 3.9+, Java 8 or 11, and dependencies installed:

```bash
source .venv/bin/activate
python3 spark_pipeline.py
```

The pipeline downloads/streams the data, trains and evaluates the models, generates analysis artifacts, and saves the selected model. To serve existing artifacts without retraining:

```bash
source .venv/bin/activate
PORT=5001 python3 api.py
```

Then open `http://localhost:5001`. Use another available port if 5001 is occupied.

## Conclusion

The project demonstrates a complete local Spark ML workflow from dataset ingestion and classifier comparison through aspect and complaint analysis to API-backed dashboard delivery. Logistic Regression provided the strongest F1 score in the verified run, while error analysis highlighted negation as an area for improvement. The ABSA and topic results add useful exploratory signals, with their lexicon- and model-based limitations kept in view.
