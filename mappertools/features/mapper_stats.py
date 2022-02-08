import pandas

import mappertools.features.core as mfc




def compute_centrality_measures(nxgraph, unique_entities, centrality_functions, aggregation_functions,
                                query_data='unique_members'):

    centrality_names = [ cen_fun.__name__ for cen_fun in centrality_functions ]
    centrality_dicts = [ cen_fun(nxgraph) for cen_fun in centrality_functions ]

    columns = []
    for cen_name in centrality_names:
        columns.append(cen_name)
        for agg_fun in aggregation_functions:
            columns.append(cen_name + "_" + agg_fun.__name__)

    ans = pandas.DataFrame(index=unique_entities, columns=columns)

    for entity in unique_entities:
        entity_nodes = list(mfc.get_nodes_containing_entity(nxgraph, entity, query_data))

        for cen_name, cen_dict in zip(centrality_names, centrality_dicts):
            entity_centralities = [cen_dict[node] for node in entity_nodes]

            # as-is output
            ans.loc[entity, cen_name] = entity_centralities

            if len(entity_centralities) == 0:
                continue

            for agg_fun in aggregation_functions:
                key = cen_name + "_" + agg_fun.__name__
                ans.loc[entity, key] = agg_fun(entity_centralities)

    return ans
