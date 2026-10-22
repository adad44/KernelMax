"""Intentional corruption for checking parent-owned numerical grading."""


def run(experts, x, selected_expert_ids):
    return experts(x, selected_expert_ids) + 100
