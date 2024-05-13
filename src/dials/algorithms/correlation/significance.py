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
        clusters = []
        for cluster in self.linkage:
            clusters.append([i - 1 for i in self.linkage[cluster]["datasets"]])
        self.completed_clusters = {}
        for ids in clusters:
            is_mixed = self.expected_significance(ids)
            new_data, sub_clusters = self.id_sub_clusters(ids)
            if len(new_data) == 1:
                array_1 = self._merge_intensities(
                    [self.unmerged_datasets[new_data[0]].deep_copy()]
                )[0]
                array_2 = self.completed_clusters[tuple(sub_clusters[0])].deep_copy()
            elif len(new_data) == 0 and len(sub_clusters) == 1:
                array_1 = self._merge_intensities(
                    [self.unmerged_datasets[ids[0]].deep_copy()]
                )[0]
                array_2 = self._merge_intensities(
                    [self.unmerged_datasets[ids[1]].deep_copy()]
                )[0]
            elif len(sub_clusters) == 2:
                array_1 = self.completed_clusters[tuple(sub_clusters[0])].deep_copy()
                array_2 = self.completed_clusters[tuple(sub_clusters[1])].deep_copy()
            significance, pval, q, dof = self.calculate_significance(array_1, array_2)
            merged_cluster = self.merge_cluster(ids)
            self.completed_clusters[tuple(ids)] = merged_cluster
            self.cluster_significance[tuple(ids)] = [significance, pval, q, dof]
            if significance != is_mixed:
                print("Stats don't agree for cluster:")
                print(tuple(ids))
                print(self.cluster_significance[tuple(ids)])
                print(
                    f"From data ids this cluster is a mix of cows and people: {is_mixed}"
                )
        # print(self.cluster_significance)

    def expected_significance(self, ids):
        cows = list(range(0, 34))
        cows += list(range(82, 150))
        man = list(range(34, 82))
        man += list(range(150, 169))

        status = None
        for i in ids:
            if not status:
                if i in cows:
                    status = "cows"
                elif i in man:
                    status = "man"
            elif status == "cows":
                if i in man:
                    status = "mixed"
            elif status == "man":
                if i in cows:
                    status = "mixed"

        if status == "mixed":
            is_mixed = True
        else:
            is_mixed = False

        return is_mixed

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
        # try:
        int1, int2 = arr1.common_sets(arr2)
        # except:
        # print(arr1.crystal_symmetry())
        # print(arr2.crystal_symmetry())
        int1, int2 = arr1.common_sets(arr2, assert_is_similar_symmetry=False)

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
                cluster.set_info(self.unmerged_datasets[i])
            else:
                # try:
                cluster = cluster.concatenate(self.unmerged_datasets[i].deep_copy())
                # except:
                # print(i)
                # print(cluster.crystal_symmetry())
                # print(self.unmerged_datasets[i].deep_copy().crystal_symmetry())
                cluster = cluster.concatenate(
                    self.unmerged_datasets[i].deep_copy(),
                    assert_is_similar_symmetry=False,
                )

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
