import csv

import numpy as np

import graphviz
import pandas as pd
import rootpath
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from skexplain.enums.feature_type import FeatureType
from skexplain.imitation import (ClassificationDagger, ClassificationEndear,
                                 RegressionDagger, RegressionEndear)
from skexplain.utils import dataset, log, persist
from skexplain.utils.const import (BOSTON_DATASET_META,
                                   CIC_IDS_2017_DATASET_META,
                                   DIABETES_DATASET_META, IOT_DATASET_META,
                                   WINE_DATASET_META)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC

RESULTS_FILE_NAME = "{}/res/results/endear_test.csv"


def endear_test(dataset_meta, validate_dataset_path="", model=RandomForestClassifier, resampler=None, num_leaves=None, num_samples=2000, as_df=False):
    """ Test using Reinforcement Learning to extract Decision Tree from a generic Blackbox model """
    logger = log.Logger(
        "{}/res/log/{}/endear_test_{}_{}.log".format(rootpath.detect(),
                                                     dataset_meta['name'],
                                                     model.__name__,
                                                     resampler.__name__ if resampler else "Raw")
    )

    # Step 1: Load training dataset
    logger.log("#" * 10, "Dataset init", "#" * 10)
    logger.log("Reading dataset fromn CSV...")
    X, y, feature_names, numerical, categorical = dataset.read(
        dataset_meta['path'], metadata=dataset_meta, verbose=True, logger=logger, resampler=resampler, as_df=as_df)
    logger.log("Done!")

    logger.log("Splitting dataset into training and test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    logger.log("Done!")
    logger.log("#" * 10, "Done", "#" * 10)

    # Step 2: Train black-box model with loaded dataset
    logger.log("#" * 10, "Model train", "#" * 10)
    model_path = "../res/weights/{}_{}_{}.joblib".format(model.__name__,
                                                         resampler.__name__ if resampler else "Raw",
                                                         dataset_meta['name'])
    logger.log("Looking for pre-trained model: {}...".format(model_path))
    blackbox = persist.load_model(model_path)
    if not blackbox:
        logger.log("Model path does not exist.")
        logger.log("Training model: {}...".format(model))

        logger.log("y_train", y_train)
        blackbox = model()
        blackbox.fit(X_train, y_train if isinstance(y_train, pd.DataFrame) else y_train.ravel())
        logger.log("Done!")
        if model_path:
            persist.save_model(blackbox, model_path)

    logger.log("#" * 10, "Done", "#" * 10)

    logger.log("#" * 10, "Model test", "#" * 10)
    y_pred = blackbox.predict(X_test)

    blackbox_score = 0
    if dataset_meta['type'] == 'classification':
        logger.log("Blackbox model training classification report:")
        logger.log("\n{}".format(classification_report(y_test, y_pred, digits=3)))
        blackbox_score = f1_score(y_test, y_pred, average="macro")
        # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
    else:
        blackbox_score = r2_score(y_test, y_pred)
        logger.log("Blackbox model R2 score: {}".format(blackbox_score))

    logger.log("#" * 10, "Done", "#" * 10)

    if validate_dataset_path:
        # Step 2.a (optional): Test trained model with a validation dataset
        logger.log("Reading validation dataset fromn CSV...")
        X_validate, y_validate, _, _, _ = dataset.read(
            validate_dataset_path, metadata=dataset_meta, verbose=True, logger=logger)
        logger.log("Done!")

        logger.log("#" * 10, "Model validation", "#" * 10)
        y_validation_pred = blackbox.predict(X_validate)

        if dataset_meta['type'] == 'classification':
            logger.log("Blackbox model validation classification report:")
            logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
            # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
        else:
            logger.log("Blackbox model validation R2 score: {}".format(r2_score(y_validate, y_validation_pred)))

        logger.log("#" * 10, "Done", "#" * 10)

    if dataset_meta['type'] == 'classification':
        logger.log("Using Classification Endear algorithm...")
        endear = ClassificationEndear(expert=blackbox, logger=logger)
    else:
        logger.log("Using Regression Endear algorithm...")
        endear = RegressionEndear(expert=blackbox, logger=logger)

    n_components = len(dataset_meta['classes']) if 'classes' in dataset_meta else 10
    endear.fit(X, y, numeric_feat_inds=numerical, cat_feat_inds=categorical, max_iter=100, n_components=n_components,
               max_leaf_nodes=num_leaves, num_samples=num_samples, verbose=True)

    with open(RESULTS_FILE_NAME.format(rootpath.detect()), "a") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        logger.log("#" * 10, "Explanation validation", "#" * 10)
        (dt, reward, idx) = endear.explain()
        logger.log("Model explanation {} local fidelity: {}".format(idx, reward))
        dt_y_pred = dt.predict(X_test)

        dt_score = 0
        fidelity = 0
        if dataset_meta['type'] == 'classification':
            logger.log("Model explanation classification report:")
            logger.log("\n{}".format(classification_report(y_test, dt_y_pred, digits=3)))
            dt_score = f1_score(y_test, dt_y_pred, average="macro")

            logger.log("Model explanation global fidelity report:")
            logger.log("\n{}".format(classification_report(y_pred, dt_y_pred, digits=3)))
            fidelity = f1_score(y_pred, dt_y_pred, average="macro")
        else:
            dt_score = r2_score(y_test, dt_y_pred)
            fidelity = r2_score(y_pred, dt_y_pred)
            logger.log("Model explanation validation R2 score: {}".format(dt_score))
            logger.log("Model explanation global fidelity: {}".format(fidelity))

        if validate_dataset_path:
            logger.log("#" * 10, "Decision tree validation", "#" * 10)
            y_validation_pred = dt.predict(X_validate)

            if dataset_meta['type'] == 'classification':
                logger.log("Decision tree model validation classification report:")
                logger.log("\n{}".format(classification_report(y_validate, y_validation_pred, digits=3)))
                # logger.log("F1-score for test data: {}".format(f1_score(y_test, y_pred, average="macro")))
            else:
                logger.log("Decision tree model validation R2 score: {}".format(
                    r2_score(y_validate, y_validation_pred)))

        csv_writer.writerow([dataset_meta['name'], len(X), model.__name__, resampler.__name__ if resampler else "None",
                             dt.get_n_leaves(), blackbox_score, dt_score, fidelity])

        dot_data = tree.export_graphviz(dt,
                                        feature_names=feature_names,
                                        class_names=dataset_meta['classes'] if 'classes' in dataset_meta else None,
                                        filled=True,
                                        rounded=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("{}/res/img/{}/{}/dt_{}_{}".format(rootpath.detect(),
                                                        dataset_meta['name'],
                                                        "endear",
                                                        resampler.__name__ if resampler else "Raw",
                                                        num_leaves))
        logger.log("#" * 10, "Done", "#" * 10)


def main():
    """ Main block """

    # overwrites current results to start new tests, and writes first row
    # with open(RESULTS_FILE_NAME.format(rootpath.detect()), "w") as csv_file:
    #     csv_writer = csv.writer(csv_file, delimiter=",")
    #     csv_writer.writerow(['dataset', 'dataset size', 'model', 'resampler',
    #                          'num leaves', 'blackbox f1/r2', 'DT f1/r2', 'fidelity'])

    # endear_test(IOT_DATASET_META, model=RandomForestClassifier, resampler=None, num_samples=10000)
    # endear_test(IOT_DATASET_META, model=RandomForestClassifier, resampler=RandomOverSampler, num_samples=100000)
    # endear_test(IOT_DATASET_META, model=RandomForestClassifier, resampler=RandomUnderSampler, num_samples=1000)
    #
    # endear_test(IOT_DATASET_META, model=MLPClassifier, resampler=None, num_samples=100000)
    # endear_test(IOT_DATASET_META, model=MLPClassifier, resampler=RandomOverSampler, num_samples=100000)
    # endear_test(IOT_DATASET_META, model=MLPClassifier, resampler=RandomUnderSampler, num_samples=1000)

    # endear_test(WINE_DATASET_META, model=RandomForestRegressor, num_samples=1000)
    # endear_test(WINE_DATASET_META, model=RandomForestRegressor, resampler=RandomOverSampler, num_samples=3000)
    # endear_test(WINE_DATASET_META, model=RandomForestRegressor, resampler=RandomUnderSampler, num_samples=10)
    #
    # endear_test(WINE_DATASET_META, model=MLPRegressor, num_samples=1000)
    # endear_test(WINE_DATASET_META, model=MLPRegressor, resampler=RandomOverSampler, num_samples=3000)
    # endear_test(WINE_DATASET_META, model=MLPRegressor, resampler=RandomUnderSampler, num_samples=10)
    #
    # endear_test(BOSTON_DATASET_META, model=RandomForestRegressor, num_samples=500)
    # endear_test(BOSTON_DATASET_META, model=MLPRegressor, num_samples=500)

    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier, num_leaves=50, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                num_leaves=50, resampler=RandomOverSampler, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                num_leaves=50, resampler=RandomUnderSampler, num_samples=100)

    endear_test(CIC_IDS_2017_DATASET_META, model=MLPClassifier, num_leaves=50, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=MLPClassifier, num_leaves=50,
                resampler=RandomOverSampler, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=MLPClassifier,
                num_leaves=50, resampler=RandomUnderSampler, num_samples=100)

    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier, num_leaves=100, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                num_leaves=100, resampler=RandomOverSampler, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                num_leaves=100, resampler=RandomUnderSampler, num_samples=100)

    endear_test(CIC_IDS_2017_DATASET_META, model=MLPClassifier, num_leaves=100, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=MLPClassifier, num_leaves=100,
                resampler=RandomOverSampler, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=MLPClassifier, num_leaves=100,
                resampler=RandomUnderSampler, num_samples=100)

    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier, num_leaves=200, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                num_leaves=200, resampler=RandomOverSampler, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                num_leaves=200, resampler=RandomUnderSampler, num_samples=100)

    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier,
                resampler=RandomOverSampler, num_samples=100000)
    endear_test(CIC_IDS_2017_DATASET_META, model=RandomForestClassifier, resampler=RandomUnderSampler, num_samples=100)


if __name__ == "__main__":
    main()
