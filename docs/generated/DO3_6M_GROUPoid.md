# DO3_to_6M: exact discrete CT report

Correspondence convention: `u_M = C_m_from_a u_A`.

## C_m_from_a
```text
⎡0   1    1  ⎤
⎢            ⎥
⎢1   0    0  ⎥
⎢            ⎥
⎣0  1/3  -1/3⎦
```

Parent group order: 48
Product group order: 4
|H_C| = 4
N_C = 12
N_operators = 8
Double-coset sizes = [4, 4, 4, 4, 8, 8, 8, 8]
Burnside cross-check = 8

## H_C
```text
⎡-1  0   0 ⎤
⎢          ⎥
⎢0   -1  0 ⎥
⎢          ⎥
⎣0   0   -1⎦
```
kind=inversion, det=-1, order=2

```text
⎡-1  0  0⎤
⎢        ⎥
⎢0   1  0⎥
⎢        ⎥
⎣0   0  1⎦
```
kind=reflection, det=-1, order=2

```text
⎡1  0   0 ⎤
⎢         ⎥
⎢0  -1  0 ⎥
⎢         ⎥
⎣0  0   -1⎦
```
kind=rotation, det=1, order=2

```text
⎡1  0  0⎤
⎢       ⎥
⎢0  1  0⎥
⎢       ⎥
⎣0  0  1⎦
```
kind=identity, det=1, order=1

## Operator summaries
- O0: size=4, inverse=O0, ambivalent=True, contents={'identity': 1, 'inversion': 1, 'reflection': 1, 'rotation': 1}
- O1: size=4, inverse=O1, ambivalent=True, contents={'reflection': 2, 'rotation': 2}
- O2: size=4, inverse=O2, ambivalent=True, contents={'reflection': 2, 'rotation': 2}
- O3: size=4, inverse=O3, ambivalent=True, contents={'rotation': 2, 'rotoinversion': 2}
- O4: size=8, inverse=O4, ambivalent=True, contents={'reflection': 2, 'rotation': 4, 'rotoinversion': 2}
- O5: size=8, inverse=O6, ambivalent=False, contents={'rotation': 4, 'rotoinversion': 4}
- O6: size=8, inverse=O5, ambivalent=False, contents={'rotation': 4, 'rotoinversion': 4}
- O7: size=8, inverse=O7, ambivalent=True, contents={'reflection': 2, 'rotation': 4, 'rotoinversion': 2}

## Variant -> operator adjacency
```text
 0  1  2  3  4  4  5  5  6  6  7  7
 1  0  3  2  4  4  5  5  6  6  7  7
 2  3  0  1  5  5  4  4  7  7  6  6
 3  2  1  0  5  5  4  4  7  7  6  6
 4  4  6  6  0  1  7  7  2  3  5  5
 4  4  6  6  1  0  7  7  3  2  5  5
 6  6  4  4  7  7  0  1  5  5  2  3
 6  6  4  4  7  7  1  0  5  5  3  2
 5  5  7  7  2  3  6  6  0  1  4  4
 5  5  7  7  3  2  6  6  1  0  4  4
 7  7  5  5  6  6  2  3  4  4  0  1
 7  7  5  5  6  6  3  2  4  4  1  0
```
