from __future__ import annotations

# import copy
import logging

# import math
import sys

from scipy.stats.distributions import chi2

# from collections import OrderedDict


# from scitbx.array_family import flex

# from dials.util.multi_dataset_handling import (
# assign_unique_identifiers,
# parse_multiple_datasets,
# )

logger = logging.getLogger("dials.algorithms.correlation.analysis")


class ClusterSignificance:
    def __init__(self, unmerged_datasets, linkage_dict):
        self.linkage = linkage_dict
        self.cluster_significance = {}
        self.unmerged_datasets = unmerged_datasets
        self.clusters = []
        for cluster in self.linkage:
            self.clusters.append([i - 1 for i in self.linkage[cluster]["datasets"]])
        self.completed_clusters = {}
        for ids in self.clusters:
            new_data, sub_clusters = self.id_sub_clusters(ids)
            if len(new_data) == 1:
                array_1 = self._merge_intensities(
                    [self.unmerged_datasets[new_data[0]]]
                )[0]
                array_2 = self.completed_clusters[tuple(sub_clusters[0])]
            elif len(new_data) == 0 and len(sub_clusters) == 1:
                array_1 = self._merge_intensities([self.unmerged_datasets[ids[0]]])[0]
                array_2 = self._merge_intensities([self.unmerged_datasets[ids[1]]])[0]
            elif len(sub_clusters) == 2:
                array_1 = self.completed_clusters[tuple(sub_clusters[0])]
                array_2 = self.completed_clusters[tuple(sub_clusters[1])]
            significance, pval, q, dof = self.calculate_significance(array_1, array_2)
            merged_cluster = self.merge_cluster(ids)
            self.completed_clusters[tuple(ids)] = merged_cluster
            self.cluster_significance[tuple(ids)] = [significance, pval, q, dof]
        self.significant_clusters = self.determine_significant_clusters()
        for i in self.significant_clusters:
            print(i)

    def id_sub_clusters(self, ids):
        if len(ids) > 2:
            sub_clusters = []
            final_sub_clusters = []
            new_dataset = []
            for i in self.completed_clusters.keys():
                if len([j for j in ids if j not in i]) == 1:
                    new_dataset = [j for j in ids if j not in i]
                if len([j for j in ids if j in i]) > 0:
                    sub_clusters.append([j for j in ids if j in i])

            if len(new_dataset) == 1:
                for k in sub_clusters:
                    joined = sorted(k + new_dataset)
                    if joined == ids:
                        final_sub_clusters.append(k)
            else:
                for m in sub_clusters:
                    for n in sub_clusters:
                        joined = sorted(m + n)
                        if joined == ids:
                            final_sub_clusters.append(m)
                            final_sub_clusters.append(n)
                final_sub_clusters = final_sub_clusters[0:2]

            if len(final_sub_clusters) == 0:
                logger.info("something has gone wrong with identifying sub clusters...")
                sys.exit()
        else:
            new_dataset = []
            final_sub_clusters = [[]]

        return new_dataset, final_sub_clusters

    def calculate_significance(self, arr1, arr2):
        # unsure if need, but not always there so doing just in case for now
        arr1.is_xray_intensity_array()
        arr2.is_xray_intensity_array()

        # Do need this
        arr1 = arr1.customized_copy(crystal_symmetry=arr2.crystal_symmetry())

        int1, int2 = arr1.common_sets(arr2)

        difference = int1.data() - int2.data()
        difference_sigmas = (int1.sigmas() ** (2) + int2.sigmas() ** (2)) ** 1 / 2
        q = 0
        dof = len(difference)
        for i, j in zip(difference, difference_sigmas):
            z = i / j
            z2 = z**2
            q += z2
        p_value = chi2.sf(q, dof)
        significance = 0.05
        if p_value < significance:
            significant_cluster = True
        else:
            significant_cluster = False
        return significant_cluster, p_value, q, dof

    def merge_cluster(self, ids):
        cluster = None
        for i in ids:
            if not cluster:
                cluster = self.unmerged_datasets[i].deep_copy()
            else:
                data = self.unmerged_datasets[i].customized_copy(
                    crystal_symmetry=cluster.crystal_symmetry()
                )
                cluster = cluster.concatenate(data)

        merged_cluster = self._merge_intensities([cluster])

        return merged_cluster[0]

    def _merge_intensities(self, datasets: list) -> list:
        """
        Merge intensities and elimate systematically absent reflections.

        Args:
            datasets(list): list of cctbx.miller.array objects
        Returns:
            datasets_sys_absent_eliminated(list): list of merged cctbx.miller.array objects
        """
        individual_merged_intensities = []
        for unmerged in datasets:
            individual_merged_intensities.append(
                unmerged.merge_equivalents().array().set_info(unmerged.info())
            )
        datasets_sys_absent_eliminated = [
            d.eliminate_sys_absent(integral_only=True).primitive_setting()
            for d in individual_merged_intensities
        ]

        return datasets_sys_absent_eliminated

    def determine_significant_clusters(self):
        significant_clusters = []
        temp = []
        used_datasets = []
        for i in self.cluster_significance:
            # THIS THING NEED TO ITERATE THROUGH AND SEE AS GO UP DENDROGRAM WHERE CLUSTER BECOMES SIGNIFICANTLY DIFFERENT, AND THEN NO LONGER CONSIDER THOSE DATASETS!!!
            if self.cluster_significance[i][0]:
                cluster_1 = []
                cluster_2 = []
                import itertools

                combinations = list(itertools.combinations(self.clusters, 2))
                for j in combinations:
                    if sorted(j[0] + j[1]) == list(i):
                        cluster_1 = j[0]
                        cluster_2 = j[1]
                if len(cluster_1) == 0:
                    # need this incase staircase and only one dataset is added
                    for k in self.clusters:
                        intersection = set(i).intersection(set(k))
                        target_length = len(i) - 1
                        if len(intersection) == target_length:
                            cluster_1 = sorted(intersection)
                            cluster_2 = [p for p in i if p not in cluster_1]
                if len(cluster_1) == 0:
                    # This happens if the overall cluster has length of 2 - ie no sub clusters
                    cluster_1 = [i[0]]
                    cluster_2 = [i[1]]
                temp.append([cluster_1, cluster_2])
        for pair in temp:
            # This makes sure that if larger clusters are made from smaller clusters does not output as significant (take this out later if want)
            c1 = pair[0]
            c2 = pair[1]
            keep_c1 = True
            keep_c2 = True
            for d1 in c1:
                if d1 in used_datasets:
                    keep_c1 = False
            for d2 in c2:
                if d2 in used_datasets:
                    keep_c2 = False
            if keep_c1 and len(c1) > 1:
                used_datasets = used_datasets + c1
                significant_clusters.append(c1)
            if keep_c2 and len(c2) > 1:
                used_datasets = used_datasets + c2
                significant_clusters.append(c2)

        return significant_clusters
