import os

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, \
    OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def _process_students(X):
    """Create new features linked to the pupils"""

    # average class size
    X['average_class_size'] = X['Nb élèves'] / X['Nb divisions']
    # percentage of pupils in the general stream
    X['percent_general_stream'] = \
        X['Nb 6èmes 5èmes 4èmes et 3èmes générales'] / X['Nb élèves']
    # percentage of pupils in an european or international section
    X['percent_euro_int_section'] = \
        X['Nb 6èmes 5èmes 4èmes et 3èmes générales sections européennes et internationales'] / X['Nb élèves'] # noqa
    # percentage of pupils doing Latin or Greek
    sum_global_5_to_3 = \
        X['Nb 5èmes'] + X['Nb 4èmes générales'] + X['Nb 3èmes générales']
    X['percent_latin_greek'] = \
        X['Nb 5èmes 4èmes et 3èmes générales Latin ou Grec'] / sum_global_5_to_3 # noqa
    # percentage of pupils that are in a SEGPA class
    X['percent_segpa'] = X['Nb SEGPA'] / X['Nb élèves']

    return np.c_[
        X['average_class_size'].values,
        X['percent_general_stream'].values,
        X['percent_euro_int_section'].values,
        X['percent_latin_greek'].values,
        X['percent_segpa'].values
    ]


def _merge_naive(X):

    # read the database with the city information
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    cities_data = pd.read_csv(filepath, index_col=0)
    # merge the two databases at the city level
    df = pd.merge(
        X, cities_data, left_on='Commune et arrondissement code',
        right_on='insee_code', how='left'
    )
    keep_col_cities = [
        'population', 'SUPERF', 'med_std_living', 'poverty_rate',
        'unemployment_rate'
    ]
    # fill na by taking the average value at the departement level
    for col in keep_col_cities:
        if cities_data[col].isna().sum() > 0:
            df[col] = df[['Département code', col]]. \
                groupby('Département code'). \
                transform(lambda x: x.fillna(x.mean()))

    return df[keep_col_cities]


def get_estimator():

    students_col = [
        'Nb élèves', 'Nb divisions', 'Nb 6èmes 5èmes 4èmes et 3èmes générales',
        'Nb 6èmes 5èmes 4èmes et 3èmes générales sections européennes et internationales', # noqa
        'Nb 5èmes', 'Nb 4èmes générales', 'Nb 3èmes générales',
        'Nb 5èmes 4èmes et 3èmes générales Latin ou Grec', 'Nb SEGPA'
    ]
    num_cols = [
        'Nb élèves', 'Nb 3èmes générales', 'Nb 3èmes générales retardataires',
        "Nb 6èmes provenant d'une école EP"
    ]
    cat_cols = [
        'Appartenance EP', 'Etablissement sensible', 'CATAEU2010',
        'Situation relative à une zone rurale ou autre'
    ]
    merge_col = [
        'Commune et arrondissement code', 'Département code'
    ]
    drop_cols = [
        'Name', 'Coordonnée X', 'Coordonnée Y', 'Commune code', 'City_name',
        'Commune et arrondissement code', 'Commune et arrondissement nom',
        'Département nom', 'Académie nom', 'Région nom', 'Région 2016 nom',
        'Longitude', 'Latitude', 'Position'
    ]

    numeric_transformer = Pipeline(steps=[
        ('scale', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ])
    students_transformer = FunctionTransformer(
        _process_students, validate=False
    )
    students_transformer = make_pipeline(
                students_transformer, SimpleImputer(strategy='mean'),
                StandardScaler()
            )
    merge_transformer = FunctionTransformer(_merge_naive, validate=False)
    merge_transformer = make_pipeline(
        merge_transformer, SimpleImputer(strategy='mean')
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
            ('students', students_transformer, students_col),
            ('merge', merge_transformer, merge_col),
            ('drop cols', 'drop', drop_cols),
        ], remainder='passthrough')  # remainder='drop' or 'passthrough'

    regressor = RandomForestRegressor(
            n_estimators=5, max_depth=50, max_features=10
        )

    pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier', regressor)
    ])

    return pipeline
