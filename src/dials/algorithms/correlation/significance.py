from __future__ import annotations

import logging

from scipy.cluster import hierarchy
from scipy.stats.distributions import chi2

from dxtbx.model import ExperimentList

from dials.algorithms.correlation.plots import linkage_matrix_to_dict
from dials.algorithms.scaling.algorithm import ScalingAlgorithm
from dials.util.filter_reflections import filtered_arrays_from_experiments_reflections

logger = logging.getLogger("dials.algorithms.correlation.analysis")


class HelpfulCluster(hierarchy.ClusterNode):
    def __init__(self, original_node, datasets):
        self.id = original_node.id
        self.left = original_node.left
        self.right = original_node.right
        self.count = original_node.count
        self.dist = original_node.dist
        self.data_ids = datasets
        self.merged_array = None
        self.significance = False
        self.pval = 0
        self.dof = 0
        self.q = 0
        self.key_cluster = False
        self.is_reference = False


class ClusterSignificance:
    def __init__(
        self, unmerged_datasets, linkage_dict, experiments, reflections, params
    ):
        self.linkage = linkage_dict
        self.experiments = experiments
        self.reflections = reflections
        self.params = params
        min_datasets_per_key_cluster = 2  # keep this at at-least 1 to be sensible

        linkage_with_datasets = linkage_matrix_to_dict(self.linkage)
        clusters = hierarchy.to_tree(self.linkage, rd=True)
        self.nice_clusters = []
        for i in clusters[1]:
            if not i.is_leaf():
                for j in linkage_with_datasets:
                    if linkage_with_datasets[j]["height"] == i.dist:
                        data = [i - 1 for i in linkage_with_datasets[j]["datasets"]]
            else:
                data = [i.id]

            new = HelpfulCluster(i, data)

            if not i.is_leaf():
                for k in self.nice_clusters:
                    if k.id == i.left.id:
                        new.left = k
                    if k.id == i.right.id:
                        new.right = k

            self.nice_clusters.append(new)

        # reorder clusters in the list because not always in a smart order

        self.nice_clusters.sort(key=lambda x: len(x.data_ids), reverse=False)

        self.unmerged_datasets = unmerged_datasets

        for cluster in self.nice_clusters:
            if cluster.is_leaf():
                cluster.merged_array = self._merge_intensities(
                    [self.unmerged_datasets[cluster.id]]
                )[0]

            else:
                if cluster.left.significance and cluster.right.significance:
                    cluster.significance = True

                elif cluster.left.significance and not cluster.right.significance:
                    cluster.significance = True
                    if len(cluster.right.data_ids) >= min_datasets_per_key_cluster:
                        cluster.right.key_cluster = True

                elif cluster.right.significance and not cluster.left.significance:
                    cluster.significance = True
                    if len(cluster.left.data_ids) >= min_datasets_per_key_cluster:
                        cluster.left.key_cluster = True

                else:
                    array_1, array_2 = self.scale_and_merge_clusters(
                        cluster.left.data_ids, cluster.right.data_ids
                    )
                    (
                        cluster.significance,
                        cluster.pval,
                        cluster.q,
                        cluster.dof,
                    ) = self.calculate_significance(array_1, array_2)
                    if cluster.significance:
                        if len(cluster.left.data_ids) >= min_datasets_per_key_cluster:
                            cluster.left.key_cluster = True
                        if len(cluster.right.data_ids) >= min_datasets_per_key_cluster:
                            cluster.right.key_cluster = True

        for cluster in self.nice_clusters:
            if cluster.key_cluster:
                print(cluster.data_ids)

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
        significance = 0.05  # 0.000001
        if p_value < significance:
            significant_cluster = True
        else:
            significant_cluster = False
        return significant_cluster, p_value, q, dof

    def scale_and_merge_clusters(self, ids1, ids2):

        temp_experiments = []
        temp_reflections = []
        idx1 = []
        idx2 = []
        for idx, i in enumerate(ids1):
            temp_experiments.append(self.experiments[i])
            temp_reflections.append(self.reflections[i])
            idx1.append(idx)

        for idx, i in enumerate(ids2):
            temp_experiments.append(self.experiments[i])
            temp_reflections.append(self.reflections[i])
            idx2.append(idx)

        idx2 = [i + len(idx1) for i in idx2]

        temp_experiments = ExperimentList(temp_experiments)

        self.params.weighting.error_model.error_model_group = [idx1, idx2]
        self.params.weighting.error_model.grouping = "grouped"
        self.params.scaling_options.full_matrix = False

        scale = ScalingAlgorithm(self.params, temp_experiments, temp_reflections)
        scale.run()

        """
        ######### Put these lines back in and delete above stuff to compare against not individual scaled behaviour
        temp_experiments = self.experiments
        temp_reflections = self.reflections
        idx1 = ids1
        idx2 = ids2
        #########
        """

        datasets = filtered_arrays_from_experiments_reflections(
            temp_experiments,
            temp_reflections,
            outlier_rejection_after_filter=False,
            partiality_threshold=self.params.partiality_threshold,
        )

        array_1 = None
        array_2 = None

        for i in idx1:
            if not array_1:
                array_1 = datasets[i].deep_copy()
            else:
                data = datasets[i].customized_copy(
                    crystal_symmetry=array_1.crystal_symmetry()
                )
                array_1 = array_1.concatenate(data)

        merged_array_1 = self._merge_intensities([array_1])

        for i in idx2:
            if not array_2:
                array_2 = datasets[i].deep_copy()
            else:
                data = datasets[i].customized_copy(
                    crystal_symmetry=array_2.crystal_symmetry()
                )
                array_2 = array_2.concatenate(data)

        merged_array_2 = self._merge_intensities([array_2])

        return merged_array_1[0], merged_array_2[0]

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
