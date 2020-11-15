import collections
import logging
import pickle
from pathlib import Path
from pandas import DataFrame
import numpy
import seaborn
from matplotlib import font_manager, pyplot
from matplotlib.collections import QuadMesh
from nltk import NaiveBayesClassifier, corpus, word_tokenize
from nltk.classify import accuracy
from nltk.metrics import ConfusionMatrix
from nltk.metrics.scores import precision, recall
from pandas import read_csv
from tqdm import tqdm


class DialogueActClassifierFactory:
    """Factory to create a classifier for dialogue act classification.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.clf = None

    def get_classifier(self, classifier_file: Path, test_set_percentage: int) -> NaiveBayesClassifier:
        """Train the classifier and persist the model to the specified file, or load from an existing model.

        Args:
            classifier_file (Path): A Path object that points to the trained classifier .pickle file.
            test_set_percentage (int): The percentage of labeled NPS Chat corpus to be used as the test set (remainder will be used as the train set).

        Returns:
            NaiveBayesClassifier: Trained classifier.
        """
        if self.clf != None:
            return self.clf

        if classifier_file.is_file():
            with open(classifier_file, mode='rb') as f:
                self.clf = pickle.load(f)
                self.logger.info('Loaded trained dialogue act classifier.')
            _, _, self.test_set = self.__get_featuresets(test_set_percentage)
        else:
            self.logger.info('Training dialogue act classifier.')
            self.clf, self.test_set = self.__train(test_set_percentage)

            with open(classifier_file, mode='wb') as f:
                pickle.dump(self.clf, f)
                self.logger.info('Saved trained dialogue act classifier.')

        return self.clf

    def classify(self, dialogue: str) -> str:
        """Classify the given featureset.

        Args:
            dialogue (str): A sentence, a passage.

        Returns: 
            str: The dialogue act type.
        """
        unlabeled_data_features = self.__dialogue_act_features(dialogue)
        return self.clf.classify(unlabeled_data_features)

    def classify_prc_csv_file(self, prc_csv_file: Path) -> Path:
        """Classify the given Pull Request Comments .csv file.

        Args:
            prc_csv_file (Path): A Path object that points to the Pull Request Comments .csv file.

        Returns:
            Path: The file path of the output file.
        """
        classified_csv_file = Path(prc_csv_file.absolute().as_posix().replace('.csv', '_dac_classified.csv'))

        if classified_csv_file.exists():
            self.logger.info(f'Output file already exists, stop further processing: {classified_csv_file}')
            return classified_csv_file

        data_frame = read_csv(prc_csv_file)
        tqdm.pandas(desc='Classifying Dialogue Act')
        data_frame['dialogue_act_classification_ml'] = data_frame.progress_apply(
            lambda row: self.classify(row['body']),
            axis='columns'
        )
        data_frame.to_csv(classified_csv_file, index=False, header=True, mode='w')
        self.logger.info(f'Dialogue Act Classification completed, output file: {classified_csv_file}')
        self.__classification_report()

        return classified_csv_file

    def __dialogue_act_features(self, dialogue: str) -> dict:
        features = {}
        for word in word_tokenize(dialogue):
            features['contains({})'.format(word.lower())] = True
        return features

    def __train(self, test_set_percentage: int):
        featuresets, train_set, test_set = self.__get_featuresets(test_set_percentage)
        self.logger.info(
            f'Size of feature set: {len(featuresets)}, train on {len(train_set)} instances, test on {len(test_set)} instances.')

        # Train the dialogue act classifier.
        return NaiveBayesClassifier.train(train_set), test_set

    def __classification_report(self):
        """Prints classifier accuracy, precisions and recalls.
        """
        self.logger.info(f'Accuracy: {self.get_accuracy()}')

        precisions, recalls = self.get_precision_and_recall()
        for label in precisions.keys():
            self.logger.info(f'{label} - precision: {precisions[label]}, recall: {recalls[label]}')

    def get_accuracy(self):
        """Returns the Accuracy of the Dialogue Act Classifier.

        Returns:
            float: Accuracy.
        """
        return accuracy(self.clf, self.test_set)

    def get_confusion_matrix(self):
        """Returns the confusion matrix for the Dialogue Act Classifier.

        """
        refsets = []
        testsets = []

        for _, (features, class_label) in enumerate(self.test_set):
            refsets.append(class_label)
            observed = self.clf.classify(features)
            testsets.append(observed)

        return ConfusionMatrix(refsets, testsets)

    def print_confusion_matrix(self):
        """Produce a heatmap of the confusion matrix for the Dialogue Act Classifier.
        """
        cm = self.get_confusion_matrix()
        df_cm = DataFrame(cm._confusion, index=cm._values, columns=cm._values)
        self.__pretty_plot_confusion_matrix(df_cm)

    def get_precision_and_recall(self):
        """Returns the Precision and Recall for each class label of the Dialogue Act Classifier.

        Returns:
            tuple: (
                dict: A dictionary of the class labels and the corresponding precision.
                dict: A dictionary of the class labels and the corresponding recall.
            )
        """
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        precisions = dict()
        recalls = dict()

        for i, (features, class_label) in enumerate(self.test_set):
            refsets[class_label].add(i)
            observed = self.clf.classify(features)
            testsets[observed].add(i)

        for class_label in refsets:
            precisions[class_label] = precision(refsets[class_label], testsets[class_label])
            recalls[class_label] = recall(refsets[class_label], testsets[class_label])

        return precisions, recalls

    def __get_featuresets(self, test_set_percentage: int):
        # Extract the labeled basic messaging data.
        posts = corpus.nps_chat.xml_posts()

        # Construct the train and test data by applying the feature extractor to each post, and create a new classifier.
        featuresets = [(self.__dialogue_act_features(post.text), post.get('class'))
                       for post in posts]
        test_set_size = int(len(featuresets) * test_set_percentage / 100)
        train_set, test_set = featuresets[test_set_size:], featuresets[:test_set_size]

        return featuresets, train_set, test_set

    # Adopted the following from https://github.com/wcipriano/pretty-print-confusion-matrix/blob/master/confusion_matrix_pretty_print.py.
    def __get_new_fig(self, fn, figsize=[9, 9]):
        """ Init graphics """
        fig1 = pyplot.figure(fn, figsize)
        ax1 = fig1.gca()  # Get Current Axis
        ax1.cla()  # clear existing plot
        return fig1, ax1

    def __configcell_text_and_colors(self, array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
        """
        config cell text and colors
        and return text elements to add and to dell
        @TODO: use fmt
        """
        text_add = []
        text_del = []
        cell_val = array_df[lin][col]
        tot_all = array_df[-1][-1]
        per = (float(cell_val) / tot_all) * 100
        curr_column = array_df[:, col]
        ccl = len(curr_column)

        # last line  and/or last column
        if(col == (ccl - 1)) or (lin == (ccl - 1)):
            #tots and percents
            if(cell_val != 0):
                if(col == ccl - 1) and (lin == ccl - 1):
                    tot_rig = 0
                    for i in range(array_df.shape[0] - 1):
                        tot_rig += array_df[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif(col == ccl - 1):
                    tot_rig = array_df[lin][lin]
                    per_ok = (float(tot_rig) / cell_val) * 100
                elif(lin == ccl - 1):
                    tot_rig = array_df[col][col]
                    per_ok = (float(tot_rig) / cell_val) * 100
                per_err = 100 - per_ok
            else:
                per_ok = per_err = 0

            per_ok_s = ['%.2f%%' % (per_ok), '100%'][per_ok == 100]

            # text to DEL
            text_del.append(oText)

            # text to ADD
            font_prop = font_manager.FontProperties(weight='bold', size=fz)
            text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
            
            # Hide the instances count (cell_val) and error percentage (per_err).
            # lis_txt = ['%d' % (cell_val), per_ok_s, '%.2f%%' % (per_err)]
            lis_txt = [per_ok_s]

            lis_kwa = [text_kwargs]
            dic = text_kwargs.copy()
            dic['color'] = 'g'
            lis_kwa.append(dic)
            dic = text_kwargs.copy()
            dic['color'] = 'r'
            lis_kwa.append(dic)

            # Hide the instances count (cell_val) and error percentage (per_err).
            # lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
            lis_pos = [(oText._x, oText._y)]

            for i in range(len(lis_txt)):
                newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
                # print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
                text_add.append(newText)
            # print '\n'

            # set background color for sum cells (last line and last column)
            carr = [0.27, 0.30, 0.27, 1.0]
            if(col == ccl - 1) and (lin == ccl - 1):
                carr = [0.17, 0.20, 0.17, 1.0]
            facecolors[posi] = carr

        else:
            if(per > 0):
                txt = '%s\n%.2f%%' % (cell_val, per)
            else:
                if(show_null_values == 0):
                    txt = ''
                elif(show_null_values == 1):
                    txt = '0'
                else:
                    txt = '0\n0.0%'
            oText.set_text(txt)

            # main diagonal
            if(col == lin):
                # set color of the textin the diagonal to white
                oText.set_color('w')
                # set background color in the diagonal to blue
                facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
            else:
                oText.set_color('r')

        return text_add, text_del
    #

    def __insert_totals(self, df_cm):
        """ insert total column and line (the last ones) """
        sum_col = []
        for c in df_cm.columns:
            sum_col.append(df_cm[c].sum())
        sum_lin = []
        for item_line in df_cm.iterrows():
            sum_lin.append(item_line[1].sum())
        df_cm['Precision'] = sum_lin
        sum_col.append(numpy.sum(sum_lin))
        df_cm.loc['Recall'] = sum_col
        #print ('\ndf_cm:\n', df_cm, '\n\b\n')
    #

    def __pretty_plot_confusion_matrix(self, df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
                                       lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='y'):
        """
        print conf matrix with default layout (like matlab)
        params:
            df_cm          dataframe (pandas) without totals
            annot          print text in each cell
            cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
            fz             fontsize
            lw             linewidth
            pred_val_axis  where to show the prediction values (x or y axis)
                            'col' or 'x': show predicted values in columns (x axis) instead lines
                            'lin' or 'y': show predicted values in lines   (y axis)
        """
        if(pred_val_axis in ('col', 'x')):
            xlbl = 'Predicted'
            ylbl = 'Actual'
        else:
            xlbl = 'Actual'
            ylbl = 'Predicted'
            df_cm = df_cm.T

        # create "Total" column
        self.__insert_totals(df_cm)

        # this is for print allways in the same window
        fig, ax1 = self.__get_new_fig('Conf matrix default', figsize)

        # thanks for seaborn
        ax = seaborn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                             cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

        # set ticklabels rotation
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

        # Turn off all the ticks
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        # face colors list
        quadmesh = ax.findobj(QuadMesh)[0]
        facecolors = quadmesh.get_facecolors()

        # iter in text elements
        array_df = numpy.array(df_cm.to_records(index=False).tolist())
        text_add = []
        text_del = []
        posi = -1  # from left to right, bottom to top.
        for t in ax.collections[0].axes.texts:  # ax.texts:
            pos = numpy.array(t.get_position()) - [0.5, 0.5]
            lin = int(pos[1])
            col = int(pos[0])
            posi += 1
            #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

            # set text
            txt_res = self.__configcell_text_and_colors(
                array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

            text_add.extend(txt_res[0])
            text_del.extend(txt_res[1])

        # remove the old ones
        for item in text_del:
            item.remove()
        # append the new ones
        for item in text_add:
            ax.text(item['x'], item['y'], item['text'], **item['kw'])

        #titles and legends
        ax.set_title('Confusion Matrix')
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        pyplot.tight_layout()  # set layout slim
        pyplot.show()
    #

    def __plot_confusion_matrix_from_data(self, y_test, predictions, columns=None, annot=True, cmap="Oranges",
                                          fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8, 8], show_null_values=0, pred_val_axis='lin'):
        """
            plot confusion matrix function with y_test (actual values) and predictions (predic),
            whitout a confusion matrix yet
        """
        from pandas import DataFrame
        from sklearn.metrics import confusion_matrix

        # data
        if(not columns):
            # labels axis integer:
            ##columns = range(1, len(numpy.unique(y_test))+1)
            # labels axis string:
            from string import ascii_uppercase
            columns = ['class %s' % (i) for i in list(ascii_uppercase)[0:len(numpy.unique(y_test))]]

        confm = confusion_matrix(y_test, predictions)
        cmap = 'Oranges'
        fz = 11
        figsize = [9, 9]
        show_null_values = 2
        df_cm = DataFrame(confm, index=columns, columns=columns)
        self.pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize,
                                          show_null_values=show_null_values, pred_val_axis=pred_val_axis)
    #
