import pandas as pd

from gadgit import GeneInfo, GAInfo


def debug_1(gene_info: GeneInfo):
    rank_pair = zip(list(gene_info.data_frame['GeneName']), gene_info.frontier)
    rank_pair1 = sorted(rank_pair, reverse=True, key=lambda y: y[1])
    place_list = []
    current_place = 1
    last_element = rank_pair1[0][1]
    for i, element in enumerate(rank_pair1):
        if element[1] != last_element:
            current_place = i + 1
        last_element = element[1]
        place_list.append(current_place)
    rank_pair2 = list(zip([x[0] for x in rank_pair1], place_list))

    df_rank_pair_1 = pd.DataFrame(rank_pair1, columns=['GeneName', 'Rank']).set_index('GeneName')
    df_rank_pair_2 = pd.DataFrame(rank_pair2, columns=['GeneName', 'Rank']).set_index('GeneName')

    rank_pair = pd.concat([df_rank_pair_1, df_rank_pair_2], axis=1)
    rank_pair.columns = ['Rank1', 'Rank2']
    tuples = [(x, y) for x, y in zip(rank_pair['Rank1'], rank_pair['Rank2'])]

    return rank_pair

    # print(rank_pair2)
    #
    # pd.DataFrame()
