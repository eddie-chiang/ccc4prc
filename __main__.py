import logging
from pathlib import Path

import confuse
import pandas

from dialogueactclassification import Classifier
from manuallabeling import FileGenerator
from ml import MachineLearning


def main():
    cfg = confuse.LazyConfig('ccc4prc', __name__)
    # Add overrides on top of config.yaml for the workspace.
    cfg.set_file('./config.workspace.yaml')

    # Setting up logging.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s, %(levelname)s, %(name)s, %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z',
        handlers=[logging.StreamHandler(), logging.FileHandler(filename=cfg['log_file'].as_filename(), mode='a')])
    logger = logging.getLogger('ccc4prc')
    logger.info('Program started.')

    input_result = input('Generate Manual Labelling File? (y/n): ')
    if is_yes(input_result):
        csv_file = Path(cfg['bigquery']['pull_request_comments_results_csv_file'].as_filename())

        dac_classifier = Classifier(
            Path(cfg['dialogue_act_classification']['trained_classifier_file'].as_filename()),
            cfg['dialogue_act_classification']['retrain_classifier'].get(bool),
            cfg['dialogue_act_classification']['test_set_percentage'].as_number())

        classified_csv_file = dac_classifier.classify_prc_csv_file(csv_file)
        manual_labelling_file_generator = FileGenerator()
        manual_labelling_file_generator.generate(classified_csv_file)

    input_result = input('Perform Machine Learning? (y/n): ')
    if is_yes(input_result):
        ml = MachineLearning()

        labeled_seed_excel_file = cfg['machine_learning']['labeled_seed_excel_file'].as_filename()
        dataset_dir = Path(labeled_seed_excel_file).parent
        training_dataset_file = dataset_dir / ('training_dataset.csv')
        test_dataset_file = dataset_dir / ('test_dataset.csv')

        training_dataset = pandas.DataFrame()
        if training_dataset_file.exists():
            training_dataset = pandas.read_csv(training_dataset_file)

        test_dataset = pandas.DataFrame()
        if test_dataset_file.exists():
            test_dataset = pandas.read_csv(test_dataset_file)

        if not training_dataset_file.exists() or not test_dataset_file.exists():
            sample_dataset = pandas.read_excel(io=labeled_seed_excel_file, sheet_name='Sample Dataset')
            training_dataset, test_dataset = ml.train_test_split(sample_dataset)

            addl_test_dataset = pandas.read_excel(io=labeled_seed_excel_file, sheet_name='Additional Test Dataset')          
            test_dataset = pandas.concat([test_dataset, addl_test_dataset], ignore_index=True)

            training_dataset.to_csv(training_dataset_file, header=True, index=False, mode='w')
            test_dataset.to_csv(test_dataset_file, header=True, index=False, mode='w')

        unlabeled_dataset = pandas.read_csv(cfg['machine_learning']['unlabeled_csv_file'].as_filename())

        ml.active_learn(training_dataset, training_dataset_file, test_dataset, unlabeled_dataset)

    # # Use the model to classify unlabeled data (BigQuery results from the CSV file).
    # comments = collections.defaultdict(set)
    # with open(pull_request_comments_csv_file, mode='r', encoding='utf-8') as input_csvfile:
    #     dict_reader = csv.DictReader(input_csvfile, delimiter=',')

    #     # Seek the file back to the start in order to use dict_reader again.
    #     input_csvfile.seek(0)
    #     next(dict_reader)  # Skip header row.

    #     for row in dict_reader:
    #         comments[row['comment_id']] = row['body']

    # # Topic Modelling with Latent Dirichlet Allocation.
    # # Step 1. Tokenization: Split the text into sentences and the sentences into workds. Lowercase the words and remove punctuation.
    # # Step 2. Words that have fewer than 3 characters are removed.
    # # Step 3. All stopwords are removed.
    # # Step 4. Words are lemmatized - words in third person are changed to first person and verbs in past and future tenses are changed into present.
    # # Step 5. Words are stemmed - words are reduced to their root form.

    # # Bag of words (BoW)
    # processed_comments = []
    # for comment_id, comment in comments.items():
    #     processed_comments.append(topic_modelling_preprocess(comment))
    # bow_dict = gensim.corpora.Dictionary(processed_comments)

    # # Filter out tokens that appear in less than 15 documents, or more than 0.5 documents (fraction of total corpus size, not absolute number),
    # # and then only keep the first 100,000 most frequent tokens.
    # bow_dict.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    # # For each document to report how many words and how many times those words appear.
    # bow_corpus = [bow_dict.doc2bow(doc) for doc in processed_comments]

    # # # Use Term Frequency-Inverse Document Frequency (TD-IDEF) to measure topic relevance.
    # # tfidf = gensim.models.TfidfModel()

    logger.info('Program ended.')


def is_yes(text: str):
    return text.lower() == 'y' or text.lower() == 'yes'
# def lemmatize_stemming(text):
#     stemmer = SnowballStemmer('english')
#     return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# def topic_modelling_preprocess(text):
#     result = []
#     for token in gensim.utils.simple_preprocess(text):
#         # All stopwords are removed, and words that have fewer than 3 characters are removed.
#         if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
#             result.append(lemmatize_stemming(token))
#     return result

# def mannwhitneyutest(logger):
#     from scipy.stats import mannwhitneyu
#     x_non_code_comprehension_related = [3, 2, 4, 7, 1, 4, 9, 18, 6, 1, 5, 3, 3, 8, 3, 1, 2, 30, 0, 1, 23, 3, 4, 5, 1, 1, 2, 2, 4, 36, 6, 11, 22, 9, 0, 35, 1, 0, 1, 3, 6, 13, 0, 14, 30, 0, 34, 2, 3, 14, 14, 2, 33, 1, 20, 1, 8, 18, 0, 1, 0, 2, 10, 32, 6, 8, 2, 0, 2, 1, 27, 33, 33, 11, 2, 17, 1, 1, 1, 1, 24, 5, 6, 0, 2, 69, 2, 23, 10, 3, 0, 10, 0, 8, 0, 2, 6, 2, 2, 1, 1, 12, 3, 13, 7, 23, 8, 1, 11, 13, 3, 5, 9, 2, 16, 14, 1, 1, 1, 6, 3, 10, 12, 12, 0, 4, 36, 9, 10, 8, 20, 26, 2, 2, 5, 35, 0, 7, 4, 2, 2, 4, 2, 36, 8, 16, 16, 0, 8, 1, 0, 11, 26, 2, 9, 0, 2, 2, 37, 1, 1, 11, 21, 6, 23, 4, 4, 1, 4, 8, 8, 2, 4, 0, 5, 19, 1, 6, 0, 4, 7, 1, 6, 0, 0, 42, 3, 8, 5, 9, 3, 2, 3, 2, 16, 3, 5, 10, 4, 2, 1, 0, 2, 5, 1, 2, 6, 0, 1, 11, 9, 4, 5, 2, 21, 6, 1, 0, 3, 2, 9, 73, 0, 10, 4, 1, 4, 0, 18, 8, 0, 13, 2, 9, 18, 64, 34, 12, 5, 2, 0, 4, 3, 31, 14, 2, 13, 0, 3, 3, 8, 7, 16, 3, 0, 2, 7, 1, 22, 0, 0, 4, 7, 12, 25, 11, 19, 4, 5, 0, 6, 0, 3, 8, 3, 3, 7, 3, 3, 5, 20, 2, 10, 1, 2, 11, 0, 52, 0, 0, 2, 15, 7, 1, 0, 19]
#     y_code_comprehension_related = [7, 9, 4, 1, 36, 3, 8, 9, 59, 12, 8, 2, 0, 3, 2, 1, 3, 2, 8, 6, 49, 1, 4, 2, 2, 2, 1, 0, 6, 1, 0, 5, 6, 7, 2, 21, 5, 1, 0, 1, 0, 16, 5, 3, 8, 19, 1, 0, 1, 4, 6, 9, 10, 10, 8, 10, 11, 31, 1, 0, 13, 6, 31, 9, 9, 12, 2, 0, 0, 1, 22, 3, 6, 28, 3, 2, 2, 2, 3, 0, 1, 6, 18, 36, 20, 2, 3, 6, 4]
#     statistic, pvalue = mannwhitneyu(x=x_non_code_comprehension_related, y=y_code_comprehension_related, use_continuity=True, alternative='two-sided')
#     logger.info(f'Mann-Whitney U Test, statistic: {statistic}, p-value: {pvalue}.')

# Execute only if run as a script
if __name__ == "__main__":
    main()
