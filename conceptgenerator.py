import random
class CustomLearningProblemGenerator:
    """
    Learning problem generator.
    """

    def __init__(self, knowledge_base, refinement_operator, num_problems=10, depth=2, min_length=2):
        self.kb = knowledge_base
        self.rho = refinement_operator
        self.num_problems = num_problems
        self.depth = depth
        self.min_length = min_length

    def apply_rho(self, node):
        refinements = [self.rho.getNode(i, parent_node=node) for i in
                       self.rho.refine(node, maxlength=
                       len(node) + self.min_length
                       if len(node) <= self.min_length
                       else len(node))]
        if len(refinements) > 0:
            return refinements

    def apply(self):
        root = self.rho.getNode(self.kb.thing, root=True)
        current_state = root
        path = []
        for _ in range(self.depth):
            refts = self.apply_rho(current_state)
            current_state = random.sample(refts, 1)[0] if refts else None
            if current_state is None:
                return path
            path.extend(refts)
        return path

    def __iter__(self):
        for _ in range(self.num_problems):
            yield from self.apply()