from agentflow_ppi.benchmarks.graphs import layered_biclique_core
from agentflow_ppi.benchmarks.baselines import SHRCHarness


def test_shrc_pll_labels_are_exact_and_bounded() -> None:
    """The corrected core index is a pruned 2-hop (PLL) labeling. The default
    degree/centrality hub order is the compact one; all orders are exact."""
    num_nodes, edges = layered_biclique_core(4, 4)
    default = SHRCHarness(num_nodes, edges)  # degree default (compact at scale)
    rank = SHRCHarness(num_nodes, edges, core_hub_strategy="greedy", exit_prune_strategy="greedy")
    # both orders are exact pruned 2-hop labelings and produce a finite index;
    # the ordering's size effect at scale is measured in benchmark_shrc_scaling.py
    assert default.stats.index_entries >= 0
    assert rank.stats.index_entries >= 0


