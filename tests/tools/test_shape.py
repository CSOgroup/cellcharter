from anndata import AnnData
import numpy as np
import pandas as pd
from shapely import Polygon

import cellcharter as cc


# Test for cc.tl.boundaries, that computes the topological boundaries of sets of cells.
class TestBoundaries:
    def test_boundaries(self, codex_adata: AnnData):
        cc.gr.connected_components(codex_adata, cluster_key="cluster_cellcharter", min_cells=250)
        cc.tl.boundaries(codex_adata)

        boundaries = codex_adata.uns["shape_component"]["boundary"]

        assert isinstance(boundaries, dict)

        # Check if boundaries contains all components of codex_adata
        assert set(boundaries.keys()) == set(codex_adata.obs["component"].cat.categories)

    def test_copy(self, codex_adata: AnnData):
        cc.gr.connected_components(codex_adata, cluster_key="cluster_cellcharter", min_cells=250)
        boundaries = cc.tl.boundaries(codex_adata, copy=True)

        assert isinstance(boundaries, dict)

        # Check if boundaries contains all components of codex_adata
        assert set(boundaries.keys()) == set(codex_adata.obs["component"].cat.categories)


class TestLinearity:
    def test_rectangle(self, codex_adata: AnnData):
        codex_adata.obs["rectangle"] = 1

        polygon = Polygon([(0, 0), (0, 10), (2, 10), (2, 0)])

        codex_adata.uns["shape_rectangle"] = {"boundary": {1: polygon}}
        linearities = cc.tl.linearity(codex_adata, "rectangle", copy=True)
        assert linearities[1] == 1.0

    def test_symmetrical_cross(self, codex_adata: AnnData):
        codex_adata.obs["cross"] = 1

        # Symmetrical cross with arm width of 2 and length of 5
        polygon = Polygon(
            [(0, 5), (0, 7), (5, 7), (5, 12), (7, 12), (7, 7), (12, 7), (12, 5), (7, 5), (7, 0), (5, 0), (5, 5)]
        )

        codex_adata.uns["shape_cross"] = {"boundary": {1: polygon}}
        linearities = cc.tl.linearity(codex_adata, "cross", copy=True)

        # The cross is symmetrical, so the linearity should be 0.5
        assert abs(linearities[1] - 0.5) < 0.01

    def test_thickness(self, codex_adata: AnnData):
        # The thickness of the cross should not influence the linearity
        codex_adata.obs["cross"] = 1

        # Symmetrical cross with arm width of 2 and length of 5
        polygon1 = Polygon(
            [(0, 5), (0, 6), (5, 6), (5, 11), (6, 11), (6, 6), (11, 6), (11, 5), (6, 5), (6, 0), (5, 0), (5, 5)]
        )

        # Symmetrical cross with arm width of 2 and length of 5
        polygon2 = Polygon(
            [(0, 5), (0, 7), (5, 7), (5, 12), (7, 12), (7, 7), (12, 7), (12, 5), (7, 5), (7, 0), (5, 0), (5, 5)]
        )

        codex_adata.uns["shape_cross"] = {"boundary": {1: polygon1}}
        linearities1 = cc.tl.linearity(codex_adata, "cross", copy=True)

        codex_adata.uns["shape_cross"] = {"boundary": {1: polygon2}}
        linearities2 = cc.tl.linearity(codex_adata, "cross", copy=True)

        assert abs(linearities1[1] - linearities2[1]) < 0.01


class TestRelativeComponentSize:
    def test_relative_component_size(self, codex_adata: AnnData):
        """
        Test the relative component size metric with a toy dataset.
        
        Setup:
        - Neighborhood 0: 66% of cells, one component (component 0)
        - Neighborhood 1: 33% of cells, split 50% into component 1 and 50% into component 2
        
        Expected RCS values:
        - Component 0: should be 1.0 (only component in neighborhood 0)
        - Component 1: should be 1.0 (50% of neighborhood 1, which has 2 components)
        - Component 2: should be 1.0 (50% of neighborhood 1, which has 2 components)
        """
        
        total_cells = len(codex_adata)
        
        # Calculate counts based on the specified distribution
        # 66% neighborhood 0, 33% neighborhood 1
        nbh_0_count = int(total_cells * 0.66)
        nbh_1_count = total_cells - nbh_0_count
        
        # Neighborhood 1 is split 50% into component 1 and 50% into component 2
        comp_1_count = nbh_1_count // 2
        comp_2_count = nbh_1_count - comp_1_count
        
        # Create domain (neighborhood) assignments
        domains = np.zeros(total_cells, dtype=int)
        domains[nbh_0_count:] = 1  # First 66% get domain 0, rest get domain 1
        
        # Create component assignments
        components = np.full(total_cells, -1, dtype=int)
        components[:nbh_0_count] = 0  # All cells in domain 0 get component 0
        components[nbh_0_count:nbh_0_count + comp_1_count] = 1  # First half of domain 1 gets component 1
        components[nbh_0_count + comp_1_count:] = 2  # Second half of domain 1 gets component 2
        
        # Add the assignments to the adata
        codex_adata.obs["domain"] = pd.Categorical(domains)
        codex_adata.obs["component"] = pd.Categorical(components)
        codex_adata.obs["sample"] = "test"
        
        rcs_values = cc.tl.relative_component_size_metric(
            codex_adata, 
            neighborhood_key="domain", 
            cluster_key="component",
            copy=True
        )
        
        # Component 0: only component in domain 0, so RCS = 1.0
        # Component 1: 50% of domain 1, which has 2 components, so RCS = 1.0
        # Component 2: 50% of domain 1, which has 2 components, so RCS = 1.0
        assert abs(rcs_values[0] - 1.0) < 1e-2
        assert abs(rcs_values[1] - 1.0) < 1e-2
        assert abs(rcs_values[2] - 1.0) < 1e-2

    def test_relative_component_size_unequal_distribution(self, codex_adata: AnnData):
        """
        Test RCS metric with unequal component distribution within a neighborhood.
        
        Setup:
        - Neighborhood 0: 60% of cells, one component (component 0)
        - Neighborhood 1: 40% of cells, split 75% into component 1 and 25% into component 2
        
        Expected RCS values:
        - Component 0: should be 1.0 (only component in neighborhood 0)
        - Component 1: should be 1.5 (75% of neighborhood 1, which has 2 components, so 0.75/0.5 = 1.5)
        - Component 2: should be 0.5 (25% of neighborhood 1, which has 2 components, so 0.25/0.5 = 0.5)
        """
        
        total_cells = len(codex_adata)
        
        # Calculate counts based on the specified distribution
        # 60% neighborhood 0, 40% neighborhood 1
        nbh_0_count = int(total_cells * 0.60)
        nbh_1_count = total_cells - nbh_0_count
        
        # Neighborhood 1 is split 75% into component 1 and 25% into component 2
        comp_1_count = int(nbh_1_count * 0.75)
        
        # Create domain (neighborhood) assignments
        domains = np.zeros(total_cells, dtype=int)
        domains[nbh_0_count:] = 1  # First 60% get domain 0, rest get domain 1
        
        # Create component assignments
        components = np.full(total_cells, -1, dtype=int)
        components[:nbh_0_count] = 0  # All cells in domain 0 get component 0
        components[nbh_0_count:nbh_0_count + comp_1_count] = 1  # 75% of domain 1 gets component 1
        components[nbh_0_count + comp_1_count:] = 2  # 25% of domain 1 gets component 2
        
        codex_adata.obs["domain"] = pd.Categorical(domains)
        codex_adata.obs["component"] = pd.Categorical(components)
        codex_adata.obs["sample"] = "test"
        
        rcs_values = cc.tl.relative_component_size_metric(
            codex_adata, 
            neighborhood_key="domain", 
            cluster_key="component",
            copy=True
        )
        
        # Component 0: only component in domain 0, so RCS = 1.0
        # Component 1: 75% of domain 1, which has 2 components, so RCS = 0.75/0.5 = 1.5
        # Component 2: 25% of domain 1, which has 2 components, so RCS = 0.25/0.5 = 0.5
        assert abs(rcs_values[0] - 1.0) < 1e-2
        assert abs(rcs_values[1] - 1.5) < 1e-2
        assert abs(rcs_values[2] - 0.5) < 1e-2

    def test_relative_component_size_multiple_neighborhoods(self, codex_adata: AnnData):
        """
        Test RCS metric with multiple neighborhoods and varying component distributions.
        
        Setup:
        - Neighborhood 0: 40% of cells, one component (component 0)
        - Neighborhood 1: 35% of cells, split 60% into component 1 and 40% into component 2
        - Neighborhood 2: 25% of cells, split 33% into component 3, 33% into component 4, 34% into component 5
        
        Expected RCS values:
        - Component 0: should be 1.0 (only component in neighborhood 0)
        - Component 1: should be 1.2 (60% of neighborhood 1, which has 2 components, so 0.6/0.5 = 1.2)
        - Component 2: should be 0.8 (40% of neighborhood 1, which has 2 components, so 0.4/0.5 = 0.8)
        - Component 3: should be 0.99 (33% of neighborhood 2, which has 3 components, so 0.33/0.333 ≈ 0.99)
        - Component 4: should be 0.99 (33% of neighborhood 2, which has 3 components, so 0.33/0.333 ≈ 0.99)
        - Component 5: should be 1.02 (34% of neighborhood 2, which has 3 components, so 0.34/0.333 ≈ 1.02)
        """
        total_cells = len(codex_adata)
        
        # Calculate counts based on the specified distribution
        nbh_0_count = int(total_cells * 0.40)
        nbh_1_count = int(total_cells * 0.35)
        nbh_2_count = total_cells - nbh_0_count - nbh_1_count
        
        # Neighborhood 1: 60% component 1, 40% component 2
        comp_1_count = int(nbh_1_count * 0.60)
        
        # Neighborhood 2: 33% each for components 3, 4, and 34% for component 5
        comp_3_count = int(nbh_2_count * 0.33)
        comp_4_count = int(nbh_2_count * 0.33)
        
        # Create domain (neighborhood) assignments
        domains = np.zeros(total_cells, dtype=int)
        domains[nbh_0_count:nbh_0_count + nbh_1_count] = 1  # Middle 35% get domain 1
        domains[nbh_0_count + nbh_1_count:] = 2  # Last 25% get domain 2
        
        # Create component assignments
        components = np.full(total_cells, -1, dtype=int)
        components[:nbh_0_count] = 0  # All cells in domain 0 get component 0
        components[nbh_0_count:nbh_0_count + comp_1_count] = 1  # 60% of domain 1 gets component 1
        components[nbh_0_count + comp_1_count:nbh_0_count + nbh_1_count] = 2  # 40% of domain 1 gets component 2
        components[nbh_0_count + nbh_1_count:nbh_0_count + nbh_1_count + comp_3_count] = 3  # 33% of domain 2 gets component 3
        components[nbh_0_count + nbh_1_count + comp_3_count:nbh_0_count + nbh_1_count + comp_3_count + comp_4_count] = 4  # 33% of domain 2 gets component 4
        components[nbh_0_count + nbh_1_count + comp_3_count + comp_4_count:] = 5  # 34% of domain 2 gets component 5
        
        codex_adata.obs["domain"] = pd.Categorical(domains)
        codex_adata.obs["component"] = pd.Categorical(components)
        codex_adata.obs["sample"] = "test"
        
        rcs_values = cc.tl.relative_component_size_metric(
            codex_adata, 
            neighborhood_key="domain", 
            cluster_key="component",
            copy=True
        )
        
        assert abs(rcs_values[0] - 1.0) < 1e-2  # Component 0: only component in domain 0
        assert abs(rcs_values[1] - 1.2) < 1e-2  # Component 1: 60% of domain 1 (2 components)
        assert abs(rcs_values[2] - 0.8) < 1e-2  # Component 2: 40% of domain 1 (2 components)
        assert abs(rcs_values[3] - 0.99) < 1e-2  # Component 3: 33% of domain 2 (3 components)
        assert abs(rcs_values[4] - 0.99) < 1e-2  # Component 4: 33% of domain 2 (3 components)
        assert abs(rcs_values[5] - 1.02) < 1e-2  # Component 5: 34% of domain 2 (3 components)

    def test_relative_component_size_cross_sample_domains(self, codex_adata: AnnData):
        """
        Test RCS metric when components from the same domain are distributed across different samples.
        
        Setup:
        - Domain 0: 50% of cells, split across two samples
          * Sample "BALBc-1": 30% of total cells, one component (component 0)
          * Sample "MRL-5": 20% of total cells, one component (component 1)
        - Domain 1: 50% of cells, split across two samples
          * Sample "BALBc-1": 25% of total cells, split 60/40 into components 2 and 3
          * Sample "MRL-5": 25% of total cells, split 40/60 into components 4 and 5
        
        Expected RCS values (calculated across all samples for each domain):
        - Component 0: 1.2 (60% of domain 0, 2 components, expected average 50%)
        - Component 1: 0.8 (40% of domain 0, 2 components, expected average 50%)
        - Component 2: 1.2 (15% of domain 1, 4 components, expected average 12.5%)
        - Component 3: 0.8 (10% of domain 1, 4 components, expected average 12.5%)
        - Component 4: 0.8 (10% of domain 1, 4 components, expected average 12.5%)
        - Component 5: 1.2 (15% of domain 1, 4 components, expected average 12.5%)
        """
        total_cells = len(codex_adata)
        
        # Calculate counts based on the specified distribution
        # Domain 0: 50% total (30% BALBc-1, 20% MRL-5)
        # Domain 1: 50% total (25% BALBc-1, 25% MRL-5)
        balbc_domain0_count = int(total_cells * 0.30)
        mrl_domain0_count = int(total_cells * 0.20)
        balbc_domain1_count = int(total_cells * 0.25)
        mrl_domain1_count = total_cells - balbc_domain0_count - mrl_domain0_count - balbc_domain1_count
        
        # Domain 1 in BALBc-1: 60% component 2, 40% component 3
        balbc_comp2_count = int(balbc_domain1_count * 0.60)
        
        # Domain 1 in MRL-5: 40% component 4, 60% component 5
        mrl_comp4_count = int(mrl_domain1_count * 0.40)
        
        # Create sample assignments
        samples = np.full(total_cells, "BALBc-1", dtype=object)
        samples[balbc_domain0_count:balbc_domain0_count + mrl_domain0_count] = "MRL-5"
        samples[balbc_domain0_count + mrl_domain0_count:balbc_domain0_count + mrl_domain0_count + balbc_domain1_count] = "BALBc-1"
        samples[balbc_domain0_count + mrl_domain0_count + balbc_domain1_count:] = "MRL-5"
        
        # Create domain (neighborhood) assignments
        domains = np.zeros(total_cells, dtype=int)
        domains[balbc_domain0_count + mrl_domain0_count:] = 1  # First 50% get domain 0, rest get domain 1
        
        # Create component assignments
        components = np.full(total_cells, -1, dtype=int)
        components[:balbc_domain0_count] = 0  # BALBc-1 domain 0 gets component 0
        components[balbc_domain0_count:balbc_domain0_count + mrl_domain0_count] = 1  # MRL-5 domain 0 gets component 1
        components[balbc_domain0_count + mrl_domain0_count:balbc_domain0_count + mrl_domain0_count + balbc_comp2_count] = 2  # BALBc-1 domain 1 first part gets component 2
        components[balbc_domain0_count + mrl_domain0_count + balbc_comp2_count:balbc_domain0_count + mrl_domain0_count + balbc_domain1_count] = 3  # BALBc-1 domain 1 second part gets component 3
        components[balbc_domain0_count + mrl_domain0_count + balbc_domain1_count:balbc_domain0_count + mrl_domain0_count + balbc_domain1_count + mrl_comp4_count] = 4  # MRL-5 domain 1 first part gets component 4
        components[balbc_domain0_count + mrl_domain0_count + balbc_domain1_count + mrl_comp4_count:] = 5  # MRL-5 domain 1 second part gets component 5
        
        codex_adata.obs["domain"] = pd.Categorical(domains)
        codex_adata.obs["component"] = pd.Categorical(components)
        codex_adata.obs["sample"] = pd.Categorical(samples)    

        rcs_values = cc.tl.relative_component_size_metric(
            codex_adata, 
            neighborhood_key="domain", 
            cluster_key="component",
            copy=True
        )
        
        # Domain 0 has 2 components (0 and 1) across all samples
        # Domain 1 has 4 components (2, 3, 4, 5) across all samples
        assert abs(rcs_values[0] - 1.2) < 1e-2  # Component 0: 60% of domain 0 (2 components, expected 50%)
        assert abs(rcs_values[1] - 0.8) < 1e-2  # Component 1: 40% of domain 0 (2 components, expected 50%)
        assert abs(rcs_values[2] - 1.2) < 1e-2  # Component 2: 60% of BALBc-1 domain 1 = 15% of total, domain 1 avg = 12.5%
        assert abs(rcs_values[3] - 0.8) < 1e-2  # Component 3: 40% of BALBc-1 domain 1 = 10% of total, domain 1 avg = 12.5%
        assert abs(rcs_values[4] - 0.8) < 1e-2  # Component 4: 40% of MRL-5 domain 1 = 10% of total, domain 1 avg = 12.5%
        assert abs(rcs_values[5] - 1.2) < 1e-2  # Component 5: 60% of MRL-5 domain 1 = 15% of total, domain 1 avg = 12.5%

    def test_relative_component_size_within_sample_domains(self, codex_adata: AnnData):
        """
        Test RCS metric when components from the same domain are distributed within a sample.
        
        Setup:
        - Domain 0: 50% of cells, split across two samples
          * Sample "BALBc-1": 30% of total cells, one component (component 0)
          * Sample "MRL-5": 20% of total cells, one component (component 1)
        - Domain 1: 50% of cells, split across two samples
          * Sample "BALBc-1": 25% of total cells, split 60/40 into components 2 and 3
          * Sample "MRL-5": 25% of total cells, split 40/60 into components 4 and 5
        
        Expected RCS values:
        - Component 0: should be 1.0 (only component in domain 0 of sample "BALBc-1")
        - Component 1: should be 1.0 (only component in domain 0 of sample "MRL-5")
        - Component 2: should be 1.2 (60% of domain 1 in sample "BALBc-1", which has 2 components)
        - Component 3: should be 0.8 (40% of domain 1 in sample "BALBc-1", which has 2 components)
        - Component 4: should be 0.8 (40% of domain 1 in sample "MRL-5", which has 2 components)
        - Component 5: should be 1.2 (60% of domain 1 in sample "MRL-5", which has 2 components)
        """
        total_cells = len(codex_adata)
        
        # Calculate counts based on the specified distribution
        # Domain 0: 50% total (30% BALBc-1, 20% MRL-5)
        # Domain 1: 50% total (25% BALBc-1, 25% MRL-5)
        balbc_domain0_count = int(total_cells * 0.30)
        mrl_domain0_count = int(total_cells * 0.20)
        balbc_domain1_count = int(total_cells * 0.25)
        mrl_domain1_count = total_cells - balbc_domain0_count - mrl_domain0_count - balbc_domain1_count
        
        # Domain 1 in BALBc-1: 60% component 2, 40% component 3
        balbc_comp2_count = int(balbc_domain1_count * 0.60)
        
        # Domain 1 in MRL-5: 40% component 4, 60% component 5
        mrl_comp4_count = int(mrl_domain1_count * 0.40)
        
        # Create sample assignments
        samples = np.full(total_cells, "BALBc-1", dtype=object)
        samples[balbc_domain0_count:balbc_domain0_count + mrl_domain0_count] = "MRL-5"
        samples[balbc_domain0_count + mrl_domain0_count:balbc_domain0_count + mrl_domain0_count + balbc_domain1_count] = "BALBc-1"
        samples[balbc_domain0_count + mrl_domain0_count + balbc_domain1_count:] = "MRL-5"
        
        # Create domain (neighborhood) assignments
        domains = np.zeros(total_cells, dtype=int)
        domains[balbc_domain0_count + mrl_domain0_count:] = 1  # First 50% get domain 0, rest get domain 1
        
        # Create component assignments
        components = np.full(total_cells, -1, dtype=int)
        components[:balbc_domain0_count] = 0  # BALBc-1 domain 0 gets component 0
        components[balbc_domain0_count:balbc_domain0_count + mrl_domain0_count] = 1  # MRL-5 domain 0 gets component 1
        components[balbc_domain0_count + mrl_domain0_count:balbc_domain0_count + mrl_domain0_count + balbc_comp2_count] = 2  # BALBc-1 domain 1 first part gets component 2
        components[balbc_domain0_count + mrl_domain0_count + balbc_comp2_count:balbc_domain0_count + mrl_domain0_count + balbc_domain1_count] = 3  # BALBc-1 domain 1 second part gets component 3
        components[balbc_domain0_count + mrl_domain0_count + balbc_domain1_count:balbc_domain0_count + mrl_domain0_count + balbc_domain1_count + mrl_comp4_count] = 4  # MRL-5 domain 1 first part gets component 4
        components[balbc_domain0_count + mrl_domain0_count + balbc_domain1_count + mrl_comp4_count:] = 5  # MRL-5 domain 1 second part gets component 5
        
        codex_adata.obs["domain"] = pd.Categorical(domains)
        codex_adata.obs["component"] = pd.Categorical(components)
        codex_adata.obs["sample"] = pd.Categorical(samples)
        
        rcs_values = cc.tl.relative_component_size_metric(
            codex_adata, 
            neighborhood_key="domain", 
            cluster_key="component",
            library_key="sample",
            copy=True
        )
        
        # Each component should be calculated relative to its own sample and domain
        assert abs(rcs_values[0] - 1.0) < 1e-2  # Component 0: only component in domain 0 of BALBc-1
        assert abs(rcs_values[1] - 1.0) < 1e-2  # Component 1: only component in domain 0 of MRL-5
        assert abs(rcs_values[2] - 1.2) < 1e-2  # Component 2: 60% of domain 1 in BALBc-1 (2 components)
        assert abs(rcs_values[3] - 0.8) < 1e-2  # Component 3: 40% of domain 1 in BALBc-1 (2 components)
        assert abs(rcs_values[4] - 0.8) < 1e-2  # Component 4: 40% of domain 1 in MRL-5 (2 components)
        assert abs(rcs_values[5] - 1.2) < 1e-2  # Component 5: 60% of domain 1 in MRL-5 (2 components)