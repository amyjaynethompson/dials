from __future__ import annotations

import logging

from scipy.cluster import hierarchy
from scipy.stats.distributions import chi2

from dxtbx.model import ExperimentList

from dials.algorithms.correlation.plots import linkage_matrix_to_dict
from dials.algorithms.scaling.algorithm import ScalingAlgorithm

# from dials.util.filter_reflections import filtered_arrays_from_experiments_reflections

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
                print("Key Cluster")
                print(cluster.data_ids)
                print("q-score")
                print(cluster.q)
                print("----")
            if cluster.significance and not cluster.key_cluster and cluster.q > 0:
                print("Significant Difference Detected!")
                print(cluster.data_ids)
                print("sub cluster 1")
                print(cluster.left.data_ids)
                print("sub cluster 2")
                print(cluster.right.data_ids)
                print("q-score")
                print(cluster.q)
                print("----")

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
        if p_value < self.params.significance.threshold:
            significant_cluster = True
        else:
            significant_cluster = False
        return significant_cluster, p_value, q, dof

    def scale_and_merge_clusters(self, ids1, ids2):

        from dials.algorithms.scaling.scaling_library import (
            scale_against_target,
            scaled_data_as_miller_array,
        )
        from dials.array_family import flex
        from dials.command_line.scale import phil_scope as scaling_scope

        params = scaling_scope.extract()

        # TESTING

        # params.weighting.error_model.basic.minimisation=None

        # for initial scale
        temp_experiments_1 = []
        temp_reflections_1 = []

        temp_experiments_2 = []
        temp_reflections_2 = []

        idx1 = []
        idx2 = []

        for idx, i in enumerate(ids1):
            temp_experiments_1.append(self.experiments[i])
            temp_reflections_1.append(self.reflections[i])
            idx1.append(idx)

        for idx, i in enumerate(ids2):
            temp_experiments_2.append(self.experiments[i])
            temp_reflections_2.append(self.reflections[i])
            idx2.append(idx)

        idx2 = [i + len(idx1) for i in idx2]

        e1 = ExperimentList(temp_experiments_1)
        e2 = ExperimentList(temp_experiments_2)

        individual_scale_1 = ScalingAlgorithm(params, e1, temp_reflections_1)
        individual_scale_1.run()

        individual_scale_2 = ScalingAlgorithm(params, e2, temp_reflections_2)
        individual_scale_2.run()

        # Make happy for next step
        r1 = flex.reflection_table()
        r2 = flex.reflection_table()

        for i in temp_reflections_1:
            r1.extend(i)
        for i in temp_reflections_2:
            r2.extend(i)

        # SCALE INDIVIDUALLY FIRST THEN DO THE THINGS

        a1 = scaled_data_as_miller_array([r1], e1)
        a2 = scaled_data_as_miller_array([r2], e2)
        a1 = a1.merge_equivalents().array()
        a2 = a2.merge_equivalents().array()

        # Do it this way so that the intensities have the proper error model adjustment
        r1["intensity.sum.value"] = r1["intensity.scale.value"]
        r1["intensity.sum.variance"] = r1["intensity.scale.variance"]

        r1["intensity.sum.value"] /= r1["inverse_scale_factor"]
        r1["intensity.sum.variance"] /= r1["inverse_scale_factor"] ** 2

        r2["intensity.sum.value"] = r2["intensity.scale.value"]
        r2["intensity.sum.variance"] = r2["intensity.scale.variance"]

        r2["intensity.sum.value"] /= r2["inverse_scale_factor"]
        r2["intensity.sum.variance"] /= r2["inverse_scale_factor"] ** 2

        # delete things so that the 'sum' intensity won't be corrected any further
        for k in [
            "lp",
            "qe",
            "dqe",
            "partiality",
            "intensity.prf.value",
            "intensity.prf.variance",
            "intensity.scale.value",
            "intensity.scale.variance",
            "inverse_scale_factor",
        ]:
            if k in r1:
                del r1[k]
            if k in r2:
                del r2[k]

        # reset some ids
        r1["id"] = flex.int(r1.size(), 0)
        r2["id"] = flex.int(r2.size(), 1)
        for k in list(r1.experiment_identifiers().keys()):
            del r1.experiment_identifiers()[k]
        for k in list(r2.experiment_identifiers().keys()):
            del r2.experiment_identifiers()[k]
        r1.experiment_identifiers()[0] = "0"
        r2.experiment_identifiers()[1] = "1"

        # remove the existing scaling models

        for e in e1:
            e.scaling_model = None
        for e in e2:
            e.scaling_model = None

        e1[0].identifier = "0"
        e2[0].identifier = "1"

        elist1 = ExperimentList([e1[0]])
        elist2 = ExperimentList([e2[0]])
        params.model = "KB"
        params.weighting.error_model.error_model = None

        result = scale_against_target(r1, elist1, r2, elist2, params)

        logger.info("\nFinal scaling model")
        logger.info(
            f"Scale factor: {elist1.scaling_models()[0].to_dict()['scale']['parameters'][0]}"
        )
        logger.info(
            f"B factor: {elist1.scaling_models()[0].to_dict()['decay']['parameters'][0]}"
        )

        # now calculate the significance

        a3 = scaled_data_as_miller_array([result], elist1)
        a3 = a3.merge_equivalents().array()

        logger.info("Clusters analysed:")
        logger.info(ids1)
        logger.info(ids2)

        logger.info("\nSignificance of difference of input datasets")
        res = self.calculate_significance(a1, a2)
        logger.info(
            f"significant_cluster: {res[0]}\np_value: {res[1]}\n q:{res[2]}\n dof:{res[3]}"
        )

        logger.info(
            "\nSignificance of difference of input datasets after coarse scaling"
        )
        res = self.calculate_significance(a3, a2)
        logger.info(
            f"significant_cluster: {res[0]}\np_value: {res[1]}\n q:{res[2]}\n dof:{res[3]}"
        )

        return a3, a2

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
