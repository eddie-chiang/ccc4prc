import logging
from pathlib import Path

import confuse
from pandas import DataFrame, concat, read_csv, read_excel

from classifier import DialogueActClassifierFactory
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

    dac_factory = DialogueActClassifierFactory()
    dac_factory.get_classifier(Path(cfg['dialogue_act_classification']['classifier_file'].as_filename(
    )), cfg['dialogue_act_classification']['test_set_percentage'].as_number())

    input_result = "y"  # input('Generate Manual Labelling File? (y/n): ')
    if is_yes(input_result):
        csv_file = Path(cfg['bigquery']['pull_request_comments_results_csv_file'].as_filename())

        classified_csv_file = dac_factory.classify_prc_csv_file(csv_file)
        manual_labelling_file_generator = FileGenerator()
        manual_labelling_file_generator.generate(classified_csv_file)

    input_result = "y"  # input('Perform Machine Learning? (y/n): ')
    if is_yes(input_result):
        ml = MachineLearning(dac_factory.get_classifier().labels())

        labeled_seed_excel_file = cfg['machine_learning']['labeled_seed_excel_file'].as_filename()
        dataset_dir = Path(labeled_seed_excel_file).parent
        training_dataset_file = dataset_dir / ('training_dataset.csv')
        test_dataset_file = dataset_dir / ('test_dataset.csv')

        training_dataset = DataFrame()
        if training_dataset_file.exists():
            training_dataset = read_csv(training_dataset_file)

        test_dataset = DataFrame()
        if test_dataset_file.exists():
            test_dataset = read_csv(test_dataset_file)

        if not training_dataset_file.exists() or not test_dataset_file.exists():
            sample_dataset = read_excel(io=labeled_seed_excel_file, sheet_name='Sample Dataset')
            training_dataset, test_dataset = ml.train_test_split(sample_dataset)

            addl_test_dataset = read_excel(io=labeled_seed_excel_file, sheet_name='Additional Test Dataset')
            test_dataset = concat([test_dataset, addl_test_dataset], ignore_index=True)

            training_dataset.to_csv(training_dataset_file, header=True, index=False, mode='w')
            test_dataset.to_csv(test_dataset_file, header=True, index=False, mode='w')

        unlabeled_dataset = read_csv(cfg['machine_learning']['unlabeled_csv_file'].as_filename())

        ml.active_learn(training_dataset, training_dataset_file, test_dataset, unlabeled_dataset)

    logger.info('Program ended.')


def is_yes(text: str):
    return text.lower() == 'y' or text.lower() == 'yes'


# Execute only if run as a script
if __name__ == "__main__":
    main()
