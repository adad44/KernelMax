"""Identity fixture, not an optimization. No routing/weight/tensor conversion."""


def run(experts, x, selected_expert_ids):
    return experts(x, selected_expert_ids)
