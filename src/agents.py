"""
This file contains the LogicalAgent and BayesianAgent classes that are going to
help the player throughout his gameplay (Diviners' Book).
"""
from re import findall
from itertools import combinations
import numpy as np

from constants import MAX_ATTEMPTS_EXPR_PROCCESSING

__all__ = ["LogicalAgent", "BayesianAgent"]


class LogicalAgent:
    """
    This is a prototype class for a logical agent. The processing of expressions may have
    some bugs.

    # Operators for sentences #
    Negation: ~(P)
    And: P & Q
    Or: P | Q
    Implication: P -> Q
    Equivalence: P <-> Q

    Please, try using parentheses when needed to avoid as many bugs as possible. No operator
    hierarchy has been configured.
        Example 1: A & B | C = (A & B) | C.
        Example 2: A | B & C = A | (B & C)
        Example 3: ~A -> C = (~A) -> C

    Try it out!
        print(LogicalAgent.Expression("((~A) <-> C) | B"))
        --- Output: [['B', 'A', 'C'], ['B', '~C', '~A']]

    Exceptions:
    """
    class Expression(object):
        def __init__(self, clause: str):
            self.clause = self.convert_to_cnf(clause)

        def __repr__(self):
            return str(self.clause)

        def cnf(self):
            return self.clause

        @staticmethod
        def tokenize(clause):
            """
            Tokenizes a string of logical clauses.
            """
            # Use a regular expression to split the string into tokens
            tokens = findall(r'[^-> <~|&()]+|->|<->|~|\||&|\(|\)', clause)
            return tokens

        def nest_clauses(self, clauses):
            nested_clauses = []
            i = 0
            while i < len(clauses):
                if clauses[i] == "(":
                    # Start a new nested clause
                    nested_clause = []
                    # Find the matching closing parenthesis
                    balance = 1
                    for j in range(i + 1, len(clauses)):
                        if clauses[j] == "(":
                            balance += 1
                        elif clauses[j] == ")":
                            balance -= 1
                            if balance == 0:
                                # End of the nested clause
                                nested_clauses.append(self.nest_clauses(nested_clause))
                                i = j
                                break
                        nested_clause.append(clauses[j])
                else:
                    nested_clauses.append(clauses[i])
                i += 1
            return nested_clauses

        class ExpressionProcessingError(Exception):
            """
            Exception raised when the maximum number of attempts trying to process an
            expression has been succeeded
            """
            def __init__(self, message="Succeeded maximum number of attempts to process"
                                       " the expression."):
                super().__init__(message)

        def convert_to_cnf(self, clause: str):
            # Tokenize clause
            clauses = self.tokenize(clause)
            # Convert the clauses to nested clauses
            nested_clauses = self.nest_clauses(clauses)
            # Eliminate equivalences
            no_eq_clauses = self.eliminate_equivalences(nested_clauses)
            # Eliminate implications
            no_imp_clauses = self.eliminate_implications(no_eq_clauses)
            # Apply De Morgan's laws and remove double negations
            no_neg_clauses = self.group_negations(self.eliminate_negations(no_imp_clauses))
            # Apply distributive and commutative laws.
            cnf_clause = self.simplify_clauses(no_neg_clauses)
            return cnf_clause

        def eliminate_equivalences(self, clauses):
            """
            Eliminates equivalences using the equivalences:
            p <-> q <=> (p -> q) & (q -> p)
            """
            if not clauses:
                return []

            if len(clauses) == 1:
                if isinstance(clauses[0], str):
                    return [clauses[0]]
                return [self.eliminate_equivalences(clauses[0])]
            elif clauses[1] == "<->":
                return [self.eliminate_equivalences([clauses[0]]) + ["->"] + self.eliminate_equivalences([clauses[2]]),
                        "&",
                        self.eliminate_equivalences([clauses[2]]) + ["->"] + self.eliminate_equivalences([clauses[0]])
                        ] + self.eliminate_equivalences(clauses[3:])

            return self.eliminate_equivalences([clauses[0]]) + self.eliminate_equivalences(clauses[1:])

        def eliminate_implications(self, clauses):
            """
            Eliminates implications using the equivalence: p -> q <=> ~p | q.
            """
            if not clauses:
                return []

            if len(clauses) == 1:
                if isinstance(clauses[0], str):
                    return [clauses[0]]
                return [self.eliminate_implications(clauses[0])]

            elif clauses[1] == "->":
                return ["~"] + self.eliminate_implications([clauses[0]]) + ["|"] + \
                    self.eliminate_implications(clauses[2:])
            return self.eliminate_implications([clauses[0]]) + self.eliminate_implications(clauses[1:])

        def eliminate_negations(self, clauses):
            """
            Eliminates negations using De Morgan's laws.
            """
            if not clauses:
                return []

            if len(clauses) == 1:
                if isinstance(clauses[0], str):
                    return [clauses[0]]
                return [self.eliminate_negations(clauses[0])]

            if clauses[0] == clauses[1] == "~":
                return self.eliminate_negations(clauses[2:])
            elif clauses[0] == "~" and isinstance(clauses[1], list):
                no_neg_clause = []
                for clause in clauses[1]:
                    if clause not in ["|", "&", "~"]:
                        no_neg_clause += ["~", clause]
                    elif clause == "|":
                        no_neg_clause += ["&"]
                    elif clause == "&":
                        no_neg_clause += ["|"]
                    else:
                        no_neg_clause += ["~"]
                return self.eliminate_negations([no_neg_clause]) + self.eliminate_negations(clauses[2:])
            return self.eliminate_negations([clauses[0]]) + self.eliminate_negations(clauses[1:])

        def group_negations(self, clauses):
            """
            Groups negations with symbols. Example: ["~", "A"] = ["~A"]
            """
            grouped_clauses = []
            for i in range(len(clauses)):
                clause = clauses[i]
                if isinstance(clause, list):
                    grouped_clauses.append(self.group_negations(clause))
                elif clause == "~":
                    grouped_clauses.append("~" + clauses[i+1])
                elif clauses[i-1] == "~":
                    continue
                else:
                    grouped_clauses.append(clause)
            return grouped_clauses

        def simplify_clauses(self, clauses):
            """
            Simplifies a list of clauses using distributive and commutative laws.
            """
            distributed_clauses = clauses

            def flatten_list(nested_list):
                """
                Obtains inner lists from list and extends them.
                """
                if all(type(x) == list for x in nested_list):
                    for x in nested_list:
                        for y in flatten_list(x):
                            yield y
                else:
                    yield nested_list

            def remove_duplicates(lst):
                """
                Removes duplicated lists and strings within lists.
                """
                unique_lst = []
                for sublist in lst:
                    unique_sublist = []
                    for element in sublist:
                        if element not in unique_sublist:
                            unique_sublist.append(element)
                    if unique_sublist not in unique_lst:
                        unique_lst.append(unique_sublist)
                return unique_lst

            for _ in range(MAX_ATTEMPTS_EXPR_PROCCESSING):
                distributed_clauses = self.distribute(distributed_clauses)[::-1]
                converted_clauses = list(flatten_list(self.convert_to_sets(distributed_clauses)))
                if all(isinstance(i, list) and all(isinstance(j, str) for j in i) for i in converted_clauses):
                    return remove_duplicates(converted_clauses)
            raise self.ExpressionProcessingError

        def distribute(self, clauses):
            """
            Simplifies a list of clauses using distributive.
            """
            if not clauses:
                return []

            if len(clauses) == 1:
                if isinstance(clauses[0], str):
                    return [clauses[0]]
                return [self.distribute(clauses[0])]

            if clauses[1] == "|" and isinstance(next_clause := clauses[2], list):
                prev_clause = clauses[0]
                if "&" not in next_clause:
                    clauses[2] = [prev_clause, "|"] + next_clause
                else:
                    new_clause = []
                    for clause in next_clause:
                        if clause != "&":
                            new_clause += [[prev_clause, "|", clause]]
                        else:
                            new_clause += ["&"]
                    clauses[2] = new_clause
                return self.distribute([clauses[2]]) + self.distribute(clauses[3:])
            else:
                return self.distribute([clauses[0]]) + self.distribute(clauses[1:])

        def convert_to_sets(self, clauses):
            """
            Converts to clauses to a single list of clauses separated by "&" operator.
            """
            if isinstance(clauses, str):
                return [clauses]
            elif len(clauses) == 3 and clauses[1] == '|':
                return self.convert_to_sets(clauses[0]) + self.convert_to_sets(clauses[2])
            elif len(clauses) == 2 and clauses[0] == '|':
                return self.convert_to_sets(clauses[1])
            elif len(clauses) == 1:
                return self.convert_to_sets(clauses[0])
            elif len(clauses) == 3 and clauses[1] == '&':
                return [self.convert_to_sets(clauses[0]), self.convert_to_sets(clauses[2])]

            converted_clause = []
            for item in clauses:
                if item not in ["|", "&"]:
                    converted_clause += self.convert_to_sets(item)
            return converted_clause

    def __init__(self):
        self._kb = self._create_empty_kb()

    @staticmethod
    def _create_empty_kb():
        return dict()

    def _negate_clause(self, clause):
        return self.Expression("~(" + clause + ")").cnf()

    def tell(self, *clauses):
        for clause in clauses:
            expr = self.Expression(clause).cnf()
            for clause_ in expr:
                if clause_ not in self._kb.values():
                    self._kb[len(self._kb)] = clause_

    @staticmethod
    def unify(clause1, clause2):
        result = set(clause1)
        for elem in clause2:
            if elem[0] == "~":
                if elem[1:] in result:
                    result.remove(elem[1:])
                else:
                    result.add(elem)
            else:
                if "~" + elem in result:
                    result.remove("~" + elem)
                else:
                    result.add(elem)
        return sorted(result)

    def ask(self, clause):
        # Step 0: Check if KB is not empty
        kb = self._kb.copy()

        # Step 1: Convert (KB ∧ ¬α) to Conjunctive Normal Form
        negated_clause = self._negate_clause(clause)
        for expr in negated_clause:
            kb[len(kb)] = expr

        combined = set()
        # Step 2: Keep checking to see if we can use resolution to produce a new clause
        while True:
            new = []
            all_combined = True
            for clause1_id, clause2_id in combinations(kb.keys(), 2):
                clause1, clause2 = kb[clause1_id], kb[clause2_id]
                if (clause1_id, clause2_id) in combined:
                    continue

                all_combined = False
                resolvents = self.unify(clause1, clause2)
                combined.add((clause1_id, clause2_id))

                if not resolvents:
                    return True

                if resolvents != sorted(set(clause1 + clause2)) and resolvents not in list(kb.values()):
                    new.append(resolvents)
            if all_combined:
                return False
            for expr in new:
                kb[len(kb)] = expr


class BayesianAgent:
    """
    This is bayesian agent that uses the likelihood of a Bernoulli experiment (feeling something or
    not feeling something) and a prior probability to calculate a posterior which ranges from 0 to 1.
    """
    def __init__(self, n: int, m: int, success: int, p: float = 0.90, q: float = 0.05):
        self.prior = np.full(shape=(n, m), fill_value=success / (n * m), dtype=float)
        # Probability of feeling something/nothing, based that there is something nearby
        self.likelihood0 = np.full(shape=(n, m), fill_value=success / (n * m), dtype=float)
        # Probability of feeling something/nothing, based that there is nothing nearby
        self.likelihood1 = np.full(shape=(n, m), fill_value=success / (n * m), dtype=float)
        self.prev_x, self.prev_y = None, None

        self.true_positive = p
        self.false_positive = q

    def modify_prior(self, dictionary):
        diff = 0
        for coord, value in dictionary.items():
            # Current value of the probability
            current_val = self.prior[coord[0], coord[1]]
            # Calculate the difference and subtract it from current probability
            diff += value - current_val
            # Update the element in the specified row and col
            self.prior[coord[0], coord[1]] = value
        # Find the minimum value of the remaining elements in the matrix
        remaining_elements = self.prior[np.logical_and(self.prior != 0,
                                                       ~np.isin(self.prior, list(dictionary.values())))]
        min_val = remaining_elements.min()
        # Calculate value of parts to be shared within remaining values
        min_count = sum(remaining_elements / min_val)
        # Use the minimum value to calculate the new values for the remaining elements
        self.prior[np.logical_and(self.prior != 0, ~np.isin(self.prior, list(
            dictionary.values())))] -= diff * remaining_elements / min_count

    def modify_likelihood0(self, x, y, feel: bool = True):
        self.likelihood0[x, y] = self.true_positive if feel else (1 - self.true_positive)

    def modify_likelihood1(self, x, y, feel: bool = True):
        self.likelihood1[x, y] = self.false_positive if feel else (1 - self.false_positive)

    def calculate_posterior(self, x, y):
        # posterior = likelihood * prior * constant
        prior = self.prior[x, y]
        return self.likelihood0[x, y] * prior / (self.likelihood0[x, y] * prior + self.likelihood1[x, y] * (1 - prior))
