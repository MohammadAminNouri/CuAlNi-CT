from validation.engineered.mutation_score import mutation_score


def test_every_deliberately_wrong_scientific_mutant_is_killed():
    result = mutation_score()
    survivors = [name for name, killed in result.items() if not killed]
    assert not survivors, f"validation blind spots; surviving mutants: {survivors}"
