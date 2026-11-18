"""Exact U8 ray symmetries and critical-value elimination. Run with uv run --with sympy python."""
from itertools import permutations

from sympy import I, Matrix, Poly, Rational, expand, factor, gcd, resultant, symbols

s, v = symbols("s v")
rays = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1),(0,-1,0,0),
        (0,0,0,-1),(0,0,-1,-1),(-1,0,0,1),(0,-1,0,1),(0,1,0,-1)]


def ray_actions():
    """The first four rays are the standard basis, hence determine each map."""
    ray_set = set(rays)
    for columns in permutations(rays, 4):
        images = [tuple(sum(a * col[i] for a, col in zip(ray, columns))
                        for i in range(4)) for ray in rays]
        if set(images) == ray_set:
            # Surjectivity on the lattice follows since the rays span it.
            yield tuple(rays.index(im) for im in images)


def polynomial(sign, branch, weight):
    return s**4 + (1 + weight*branch)*s**3 - (sign + weight*branch)*s - 1


def value_numerator(sign, branch, weight):
    return s**4 + 2*(1 + weight*branch)*s**3 + 2*(sign + weight*branch)*s + 1


def value_factors(weight):
    for sign in (1,-1):
        for branch in (1,-1):
            p = polynomial(sign, branch, weight)
            val = value_numerator(sign, branch, weight)
            yield Poly(resultant(p, s**2*v - val, s), v).monic()


def radical_union(polys):
    result = Poly(1, v)
    for poly in polys:
        result = result.lcm(poly.sqf_part())
    return result.monic()


def main():
    group = list(ray_actions())
    assert len(group) == 16
    orbits = {tuple(sorted({g[i] for g in group})) for i in range(len(rays))}
    assert orbits == {(0,2,6,7),(1,4,8,9),(3,5)}
    print("Full ray group order:", len(group), "orbits:", sorted(orbits))
    weights = (1, Rational(11,10))
    for weight in weights:
        ordinary = 0
        for sign in (1,-1):
            for branch in (1,-1):
                p = Poly(polynomial(sign, branch, weight), s)
                ordinary += p.sqf_part().degree()
                if weight != 1:
                    assert p.gcd(p.diff()).degree() == p.gcd(Poly(s**2+1, s)).degree() == 0
                print(weight, sign, branch, factor(p.as_expr()))
        # Exceptional locus s^2=-1 contributes two roots of b^2+(1+e)b/B+1,
        # with any b=+/-1 already counted in the ordinary list.
        extra = 0
        for sign in (1,-1):
            q = Poly(weight*s**2 + (1+sign)*s + weight, s).sqf_part()
            extra += 2*(q.degree() - q.gcd(Poly(s**2-1, s)).degree())
        print("Ordinary, additional, total:", ordinary, extra, ordinary+extra)
        assert ordinary+extra == (18 if weight == 1 else 24)
        # Exceptional values equal -2 for e=1 and -2 +/- 4i for e=-1.
        exceptional = Poly((v+2)*((v+2)**2+16), v)
        values = radical_union([*value_factors(weight), exceptional])
        print("Distinct values:", values.degree(), "critical-value polynomial:", factor(values.as_expr()))
        assert values.degree() == (10 if weight == 1 else 13)
    # Local analytic identities, with z=s+s^{-1}, b the remaining fibre variable.
    z, b = symbols("z b")
    y = b + 1/b
    assert expand((z+2)**2 - (z*z+4*z+4)) == 0
    assert ((z+(2+y)/2)**2 - (b+1)**4/(4*b*b) - (z*z+(2+y)*z)).cancel() == 0
    assert ((s+1)**4/s**2 - (s+1/s+2)**2).cancel() == 0


if __name__ == "__main__":
    main()
