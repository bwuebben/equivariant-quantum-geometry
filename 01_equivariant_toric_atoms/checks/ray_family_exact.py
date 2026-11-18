"""Exact critical-point checks for P1, V2=dP6 and V4.

Run with Python 3.
Standard library only. Arithmetic is in Z[z]/Phi_30(z), with z a primitive
30th root. No floating-point roots, tolerances, or random starts are used.
Completeness comes from the two-root classification in the manuscript,
not merely from this finite verification.
"""

import itertools
import json
from math import comb, factorial
from pathlib import Path


# Phi_30(z) = z^8 + z^7 - z^5 - z^4 - z^3 + z + 1.
MODULUS = (1, 1, 0, -1, -1, -1, 0, 1, 1)
DEGREE = 8
ZERO = (0,) * DEGREE
ONE = (1,) + (0,) * (DEGREE - 1)


def reduce_poly(coefficients):
    a = list(coefficients)
    a += [0] * max(0, DEGREE - len(a))
    for i in range(len(a) - 1, DEGREE - 1, -1):
        c = a[i]
        if c:
            for j in range(DEGREE):
                a[i - DEGREE + j] -= c * MODULUS[j]
            a[i] = 0
    return tuple(a[:DEGREE])


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(c, a):
    return tuple(c * x for x in a)


def sub(a, b):
    return add(a, scale(-1, b))


def mul(a, b):
    c = [0] * (2 * DEGREE - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return reduce_poly(c)


def power(a, n):
    answer = ONE
    for _ in range(n):
        answer = mul(answer, a)
    return answer


ZETA = reduce_poly([0, 1])
POWERS = tuple(power(ZETA, i) for i in range(30))
assert power(ZETA, 30) == ONE
assert all(power(ZETA, d) != ONE for d in (1, 2, 3, 5, 6, 10, 15))


def zeta(a):
    return POWERS[a % 30]


def cosine_sum(a):
    return add(zeta(a), zeta(-a))


def sine_difference(a):
    return sub(zeta(a), zeta(-a))


def parity(p):
    return (-1) ** sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p)))


def determinant(matrix):
    n = len(matrix)
    answer = ZERO
    for p in itertools.permutations(range(n)):
        term = ONE
        for i in range(n):
            term = mul(term, matrix[i][p[i]])
        answer = add(answer, scale(parity(p), term))
    return answer


def critical_tuples(m):
    """Classify majority root r and minority root -1/r, using exponents mod 30."""
    result = {}
    for k in range((m + 1) // 2, m + 1):
        exponent = 2 * k - m
        # In these examples exponent is 1, 3 or 5, so all roots lie in mu_30.
        roots = [a for a in range(30) if (exponent * a - 15 * (m - k)) % 30 == 0]
        assert len(roots) == exponent
        for a in roots:
            other = (15 - a) % 30
            assert other != a
            for minority in itertools.combinations(range(m), m - k):
                point = tuple(other if i in minority else a for i in range(m))
                assert point not in result
                result[point] = (k, a)
    return result


def act(point, permutation, inversion):
    # permutation maps an index i to permutation[i]; use inverse indices for
    # the left action on points, as in the geometric pullback convention.
    inverse = [0] * len(point)
    for i, image in enumerate(permutation):
        inverse[image] = i
    return tuple((inversion * point[inverse[i]]) % 30 for i in range(len(point)))


def check_family(m, name, labels):
    points = critical_tuples(m)
    expected = sum(comb(m, k) * (2 * k - m) for k in range((m + 1) // 2, m + 1))
    assert expected == m * comb(m - 1, (m - 1) // 2) == len(points)
    values = {}
    determinants = {}
    for point, (k, a) in points.items():
        assert sum(point) % 30 == 0
        critical = [sine_difference(b) for b in point]
        assert all(g == critical[-1] for g in critical)
        diagonal = [cosine_sum(b) for b in point]
        n = m - 1
        hessian = [
            [add(diagonal[i], diagonal[-1]) if i == j else diagonal[-1]
             for j in range(n)]
            for i in range(n)
        ]
        det = determinant(hessian)
        formula = scale((-1) ** (m - k) * (2 * k - m), power(cosine_sum(a), m - 1))
        assert det == formula != ZERO
        values[point] = sum_vectors(diagonal)
        determinants[point] = det
        assert values[point] == scale(2 * k - m, cosine_sum(a))

    permutations = list(itertools.permutations(range(m)))
    group = [(p, inv) for p in permutations for inv in (1, -1)]
    remaining = set(points)
    rows = []
    while remaining:
        representative = min(remaining)
        orbit = {act(representative, p, inv) for p, inv in group}
        assert orbit <= points.keys()
        remaining -= orbit
        stabilizer = [(p, inv) for p, inv in group if act(representative, p, inv) == representative]
        assert len(orbit) * len(stabilizer) == len(group)
        value = values[representative]
        assert all(values[q] == value for q in orbit)
        fixed_by_inversion = sum(tuple((-a) % 30 for a in q) == q for q in orbit)
        rows.append({
            "value": labels[value],
            "orbit_size": len(orbit),
            "stabilizer_order": len(stabilizer),
            "stabilizer_elements_with_inversion": sum(inv == -1 for _, inv in stabilizer),
            "fixed_by_central_inversion": fixed_by_inversion,
            "representative_exponents_mod_30": representative,
            "determinant_coefficients": determinants[representative],
        })
    rows.sort(key=lambda row: (row["orbit_size"], row["value"]))
    assert len({row["value"] for row in rows}) == len(rows)
    return {
        "name": name,
        "dimension": m - 1,
        "group_order": 2 * factorial(m),
        "critical_points": len(points),
        "all_critical_points_nondegenerate": True,
        "orbits": rows,
        "total_fixed_by_central_inversion": sum(row["fixed_by_central_inversion"] for row in rows),
    }


def sum_vectors(vectors):
    answer = ZERO
    for v in vectors:
        answer = add(answer, v)
    return answer


def main():
    # P1 is separate: the two signs of its single ray are one antipodal pair.
    p1 = []
    for a in (0, 15):
        assert sine_difference(a) == ZERO
        assert (-a) % 30 == a
        assert cosine_sum(a) != ZERO
        p1.append({"point": 1 if a == 0 else -1, "value": 2 if a == 0 else -2,
                   "inversion_fixed": True, "hessian": 2 if a == 0 else -2})
    dp6_labels = {scale(6, ONE): "6", scale(-3, ONE): "-3", scale(-2, ONE): "-2"}
    v4_labels = {
        scale(10, ONE): "10", scale(-6, ONE): "-6",
        scale(2, ONE): "2", scale(3, ONE): "3",
        scale(5, cosine_sum(6)): "5*(sqrt(5)-1)/2",
        scale(5, cosine_sum(12)): "-5*(sqrt(5)+1)/2",
    }
    # Certify the radical labels in the same cyclotomic field.
    sqrt5 = add(ONE, scale(2, cosine_sum(6)))
    assert mul(sqrt5, sqrt5) == scale(5, ONE)
    assert add(cosine_sum(6), cosine_sum(12)) == scale(-1, ONE)
    result = {
        "arithmetic": "exact integer coefficients modulo Phi_30",
        "p1": p1,
        "families": [
            check_family(3, "degree-six del Pezzo surface", dp6_labels),
            check_family(5, "centrally symmetric fourfold V4", v4_labels),
        ],
    }
    v4 = result["families"][1]
    tens = {r["value"]: r["fixed_by_central_inversion"] for r in v4["orbits"] if r["orbit_size"] == 10}
    assert tens == {"2": 10, "3": 0}
    output = Path(__file__).with_name("ray_family_exact_results.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "P1": "2 exact Morse points, both inversion-fixed",
        "dP6": "6 exact Morse points; orbit sizes 1,2,3",
        "V4": "30 exact Morse points; orbit sizes 1,2,2,5,10,10",
        "V4_ten_point_orbits": "central inversion fixes 10 points in one and 0 in the other",
        "result": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
