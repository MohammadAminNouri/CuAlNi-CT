# DO3_to_2H: exact discrete CT report

Correspondence convention: `u_M = C_m_from_a u_A`.

## C_m_from_a
```text
⎡0  1  1 ⎤
⎢        ⎥
⎢1  0  0 ⎥
⎢        ⎥
⎣0  1  -1⎦
```

Parent group order: 48
Product group order: 8
|H_C| = 8
N_C = 6
N_operators = 3
Double-coset sizes = [8, 8, 32]
Burnside cross-check = 3

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
⎡-1  0   0 ⎤
⎢          ⎥
⎢0   0   -1⎥
⎢          ⎥
⎣0   -1  0 ⎦
```
kind=rotation, det=1, order=2

```text
⎡-1  0  0⎤
⎢        ⎥
⎢0   0  1⎥
⎢        ⎥
⎣0   1  0⎦
```
kind=rotation, det=1, order=2

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
⎡1  0   0 ⎤
⎢         ⎥
⎢0  0   -1⎥
⎢         ⎥
⎣0  -1  0 ⎦
```
kind=reflection, det=-1, order=2

```text
⎡1  0  0⎤
⎢       ⎥
⎢0  0  1⎥
⎢       ⎥
⎣0  1  0⎦
```
kind=reflection, det=-1, order=2

```text
⎡1  0  0⎤
⎢       ⎥
⎢0  1  0⎥
⎢       ⎥
⎣0  0  1⎦
```
kind=identity, det=1, order=1

## Operator summaries
- O0: size=8, inverse=O0, ambivalent=True, contents={'identity': 1, 'inversion': 1, 'reflection': 3, 'rotation': 3}
- O1: size=8, inverse=O1, ambivalent=True, contents={'reflection': 2, 'rotation': 4, 'rotoinversion': 2}
- O2: size=32, inverse=O2, ambivalent=True, contents={'reflection': 4, 'rotation': 16, 'rotoinversion': 12}

## Variant -> operator adjacency
```text
 0  1  2  2  2  2
 1  0  2  2  2  2
 2  2  0  1  2  2
 2  2  1  0  2  2
 2  2  2  2  0  1
 2  2  2  2  1  0
```
