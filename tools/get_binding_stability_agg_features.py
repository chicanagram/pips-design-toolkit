from feature_extraction.binding import get_yasara_binding_features
from feature_extraction.stability import  get_yasara_foldx_stability_features
from feature_extraction.aggregation import  get_aggregation_features

def main():
    features_to_extract = ['binding', 'stability', 'aggregation']
    combine_feature_into_ml_dataset = False

    if 'binding' in features_to_extract:
        get_yasara_binding_features()

    if 'stability' in features_to_extract:
        get_yasara_foldx_stability_features()

    if 'aggregation' in features_to_extract:
        get_aggregation_features()

if __name__ == "__main__":
    main()