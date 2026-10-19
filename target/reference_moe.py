"""Reference wrapper for the model's unchanged selected-expert computation."""


class ReferenceMoE:
    """Expose the native per-layer expert call through the evaluator interface."""

    def __init__(self, model):
        self.model = model

    def __call__(self, layer_index, x, selected_expert_ids):
        experts = self.model.model.layers[layer_index].mlp.experts
        return experts(x, selected_expert_ids)
