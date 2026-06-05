from __future__ import annotations

#############################
# PLM model configuration   #
#############################
PLM_EMBEDDING_SIZES = {
    'esm2-33': 1280,
}


#######################################
# Atomic feature blocks loaded by name #
#######################################
BASE_FEATURE_COLUMNS = {
    'oheMT': None,
    'binding_mut': [
        'avg_epotWTRtr', 'avg_esolWTcolRtr', 'avg_esolWTvdwRtr', 'avg_surfaccWTRtr',
        'avg_epotMTRtr', 'avg_esolMTcolRtr', 'avg_esolMTvdwRtr', 'avg_surfaccMTRtr',
        'avg_epotWTCpx', 'avg_esolWTcolCpx', 'avg_esolWTvdwCpx', 'avg_surfaccWTCpx',
        'avg_epotMTCpx', 'avg_esolMTcolCpx', 'avg_esolMTvdwCpx', 'avg_surfaccMTCpx',
        'avg_epotWTLgd', 'avg_esolWTcolLgd', 'avg_esolWTvdwLgd', 'avg_surfaccWTLgd',
        'avg_DDG',
    ],
    'stability_foldx': [
        'Backbone Hbond', 'Electrostatics', 'Sidechain Hbond', 'Solvation Hydrophobic',
        'Solvation Polar', 'Van der Waals', 'Van der Waals clashes', 'backbone clash',
        'disulfide', 'energy Ionisation', 'entropy mainchain', 'entropy sidechain',
        'partial covalent bonds', 'torsional clash', 'total energy', 'Backbone Hbond.1',
        'Electrostatics.1', 'Sidechain Hbond.1', 'Solvation Hydrophobic.1',
        'Solvation Polar.1', 'Van der Waals.1', 'Van der Waals clashes.1',
        'backbone clash.1', 'disulfide.1', 'energy Ionisation.1', 'entropy mainchain.1',
        'entropy sidechain.1', 'partial covalent bonds.1', 'torsional clash.1',
        'total energy.1',
    ],
    'tango': ['Aggregation_tango', 'Beta', 'Turn', 'Helix'],
    'waltz': ['Aggregation_waltz'],
}


###########################################
# Combined multimodal feature-set aliases #
###########################################
COMPOSITE_FEATURESETS = {
    'AggStabBind-mut': (
        BASE_FEATURE_COLUMNS['binding_mut']
        + BASE_FEATURE_COLUMNS['stability_foldx']
        + BASE_FEATURE_COLUMNS['tango']
        + BASE_FEATURE_COLUMNS['waltz']
    ),
}


###########################################
# Programmatic PLM-derived feature blocks #
###########################################
def _build_plm_features(plm_name: str, embedding_size: int):
    return {
        f'{plm_name}_mut_embeddings_MT': [f'{plm_name}_{i}_MT' for i in range(1, embedding_size + 1)],
        f'{plm_name}_seq_embeddings_MT': [f'{plm_name}_{i}_MT' for i in range(1, embedding_size + 1)],
        f'{plm_name}_LLRsum_entropy': ['LLR', 'entropy'],
        f'{plm_name}_LLRsum': ['LLR'],
    }


PLM_FEATURESETS = {}
for _plm_name, _embedding_size in PLM_EMBEDDING_SIZES.items():
    PLM_FEATURESETS.update(_build_plm_features(_plm_name, _embedding_size))


########################################
# Public registries used across scripts #
########################################
feature_names_multimodal = dict(COMPOSITE_FEATURESETS)
feature_names = {
    **BASE_FEATURE_COLUMNS,
    **COMPOSITE_FEATURESETS,
    **PLM_FEATURESETS,
}


def _build_plm_component_featuresets(plm_name: str):
    return {
        f'{plm_name}_LLRsum': [f'{plm_name}_LLRsum'],
        f'{plm_name}_LLRsum_entropy': [f'{plm_name}_LLRsum_entropy'],
        f'{plm_name}_mut_embeddings_MT': [f'{plm_name}_mut_embeddings_MT'],
        f'{plm_name}_seq_embeddings_MT': [f'{plm_name}_seq_embeddings_MT'],
        f'{plm_name}_mut_embeddings_MT_LLRsum': [f'{plm_name}_mut_embeddings_MT', f'{plm_name}_LLRsum'],
        f'{plm_name}_seq_embeddings_MT_LLRsum': [f'{plm_name}_seq_embeddings_MT', f'{plm_name}_LLRsum'],
        f'AggStabBind-mut_{plm_name}_LLRsum': ['AggStabBind-mut', f'{plm_name}_LLRsum'],
        f'oheMT_{plm_name}_LLRsum': ['oheMT', f'{plm_name}_LLRsum'],
    }


########################################
# Feature-set -> component file mapping #
########################################
FEATURESET_COMPONENT_REGISTRY = {
    'AggStabBind-mut': ['AggStabBind-mut'],
}
for _plm_name in PLM_EMBEDDING_SIZES:
    FEATURESET_COMPONENT_REGISTRY.update(_build_plm_component_featuresets(_plm_name))


def resolve_component_featuresets(featureset):
    """Return the source feature blocks that should be merged for a featureset."""
    if featureset in FEATURESET_COMPONENT_REGISTRY:
        return list(FEATURESET_COMPONENT_REGISTRY[featureset])
    if featureset in feature_names:
        return [featureset]
    raise KeyError(f'Unknown featureset: {featureset}')


#####################################
# Feature-set -> storage directory  #
#####################################
FEATURESET_DIRECTORY_REGISTRY = {
    'oheMT': 'feature_extraction',
    'binding_mut': 'feature_extraction',
    'stability_foldx': 'feature_extraction',
    'tango': 'feature_extraction',
    'waltz': 'feature_extraction',
    'AggStabBind-mut': 'feature_extraction',
}
for _plm_name in PLM_EMBEDDING_SIZES:
    FEATURESET_DIRECTORY_REGISTRY.update({
        f'{_plm_name}_LLRsum': 'feature_extraction',
        f'{_plm_name}_LLRsum_entropy': 'conservation_and_distance',
        f'{_plm_name}_mut_embeddings_MT': 'feature_extraction',
        f'{_plm_name}_seq_embeddings_MT': 'feature_extraction',
    })


def get_featureset_dir(featureset):
    """Return the top-level data subdirectory for a named featureset."""
    if featureset not in FEATURESET_DIRECTORY_REGISTRY:
        raise KeyError(f'Unknown featureset: {featureset}')
    return FEATURESET_DIRECTORY_REGISTRY[featureset]


###########################################
# Feature-set panels used in evaluations  #
###########################################
def get_feature_combinations(plm_name_list, feature_combinations=None, include_nonplm_featuresets=False):
    if feature_combinations is None:
        feature_combinations = {}
    else:
        feature_combinations = dict(feature_combinations)

    feature_combinations_nonplm = {
        'AggStabBind-mut': FEATURESET_COMPONENT_REGISTRY['AggStabBind-mut'],
    }

    if include_nonplm_featuresets:
        feature_combinations.update(feature_combinations_nonplm)

    for plm_name in plm_name_list:
        feature_combinations.update(_build_plm_component_featuresets(plm_name))

    return feature_combinations
